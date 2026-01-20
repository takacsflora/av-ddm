
#%%
# fft convolution
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import FFTConvolve

class RaceModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.z_res = 0.01
        self.evidence_space = torch.arange(-50, 50, self.z_res, device=self.device)
        self.n_timesteps = 300
        self.times = torch.linspace(0, 1.5, self.n_timesteps, device=self.device)

    @staticmethod
    def gaussian(x, mu, sigma):
        """
        Creates a normalized Gaussian probability distribution.
        """
        gauss = torch.exp(-(x - mu)**2 / (2 * sigma**2))
        return gauss / torch.sum(gauss)
    
    def apply_nondectime(self, p_density, t_nd):
        """
        Applies non-decision time to the probability density function.
        """
        if t_nd > 0:
            shift = int(t_nd / self.z_res)
            p_density = torch.roll(p_density, shift)
            p_density[:shift] = 0
        return p_density
    
    def run_single_race(self, drift=0, noise=1, x0=0, bound=5):
        update_kernel = self.gaussian(self.evidence_space, drift, noise)

        # Create the mask for the absorbing boundary
        boundary_mask = torch.zeros_like(self.evidence_space)
        boundary_mask[self.evidence_space >= bound] = 1

        # History of probability density over time
        p_density_history = torch.zeros((self.n_timesteps, len(self.evidence_space)), device=self.device)

        # Probability of crossing the boundary at each time step (RT distribution)
        p_cross_boundary = torch.zeros(self.n_timesteps, device=self.device)

        # Initialize the probability density at t=0
        p_density_history[0, :] = self.gaussian(self.evidence_space, x0, self.z_res) # delta function approximation

        # Run the simulation loop
        fft_convolve = FFTConvolve(mode='same')
        for nt in range(1, self.n_timesteps):
            current_density = fft_convolve.forward(p_density_history[nt-1, :].unsqueeze(0), 
                                                   update_kernel.unsqueeze(0))

            p_cross_boundary[nt] = torch.sum(boundary_mask * current_density)

            # subtract the mass that crossed the boundary
            current_density *= (1 - boundary_mask)

            p_density_history[nt, :] = current_density

        return p_density_history, p_cross_boundary
    
    def run_two_races(self, driftL=0, driftR=0, noise=1, x0L=0, x0R=0, bound=5, t_nd=0):
        p_density_historyL, pdfL = self.run_single_race(drift=driftL, noise=noise, x0=x0L, bound=bound)
        p_density_historyR, pdfR = self.run_single_race(drift=driftR, noise=noise, x0=x0R, bound=bound)

        # now caluclate the surviving probabilities

        survivor_L = torch.sum(p_density_historyL, dim=1)
        survivor_R = torch.sum(p_density_historyR, dim=1)

        rt_distL_unscaled = torch.zeros_like(pdfL)
        rt_distL_unscaled[..., 1:] = pdfL[..., 1:] * survivor_R[..., :-1]
        rt_distR_unscaled = torch.zeros_like(pdfR)
        rt_distR_unscaled[..., 1:] = pdfR[..., 1:] * survivor_L[..., :-1]

        p_nogo = survivor_L[..., -1] * survivor_R[..., -1]
        p_go_total = 1 - p_nogo
        p_choice_total_unscaled = rt_distL_unscaled.sum(dim=-1) + rt_distR_unscaled.sum(dim=-1)
        
        scaling_factor = torch.zeros_like(p_go_total)
        mask = p_choice_total_unscaled > 1e-9
        scaling_factor[mask] = p_go_total[mask] / p_choice_total_unscaled[mask]
        
        rt_distL = rt_distL_unscaled * scaling_factor.unsqueeze(-1)
        rt_distR = rt_distR_unscaled * scaling_factor.unsqueeze(-1)

        rt_distL = self.apply_nondectime(rt_distL, t_nd)
        rt_distR = self.apply_nondectime(rt_distR, t_nd)
        
        p_nogo_final = 1 - (rt_distL.sum(dim=-1) + rt_distR.sum(dim=-1))
        
        return rt_distL, rt_distR, p_nogo_final

    def plot_pdf(self,**params):
        
        rt_distL, rt_distR, p_nogo_final = RaceModel.run_two_races(**params)

        rt_distL = rt_distL.cpu().numpy()
        rt_distR = rt_distR.cpu().numpy()
        p_nogo = p_nogo_final.cpu().numpy()
        times = self.times.cpu().numpy()

        fig,ax  = plt.subplots(figsize=(5, 6))
        ax.plot(times,rt_distR,color='k')
        ax.plot(times,-rt_distL,color='k')
        ax.vlines(x=1.5,ymin=0,ymax=p_nogo,color='k')
        ax.vlines(x=1.5,ymin=0,ymax=-p_nogo,color='k')
        
    # to be removed to another class that handles likelihoods ??
    def time_binned_likelihood(self, probs, reaction_times):
        # Ensure reaction_times is a tensor for searchsorted
        rt_tensor = torch.as_tensor(reaction_times, device=self.device)
        if rt_tensor.numel() == 0:
            return 0 # No choices, no contribution to likelihood
        rt_indices = torch.searchsorted(self.times, rt_tensor, right=True) - 1
        rt_indices = torch.clamp(rt_indices, 0, len(self.times) - 1)
        
        # Index probs correctly for time-binned likelihood
        # probs has shape [n_steps, n_choices], we need probs for each rt
        # Let's assume probs is [n_steps] for a single choice type
        log_likelihood = torch.log(probs[rt_indices] + 1e-10)
        return -torch.sum(log_likelihood)
     
class avRaceModel(RaceModel):
    def __init__(self,device='cpu'):
        super().__init__(device=device)

        self.params = nn.ParameterDict({
            'weight_vR_raw': nn.Parameter(torch.tensor(0.0)),
            'weight_vL_raw': nn.Parameter(torch.tensor(0.0)),
            'weight_aR_raw': nn.Parameter(torch.tensor(0.3)), 
            'weight_aL_raw': nn.Parameter(torch.tensor(0.3)),
            'biasL_raw': nn.Parameter(torch.tensor(0.0)),
            'biasR_raw': nn.Parameter(torch.tensor(0.0)),
            'x0L_raw': nn.Parameter(torch.tensor(0.0)),
            'x0R_raw': nn.Parameter(torch.tensor(0.0)),
            'gamma_raw': nn.Parameter(torch.tensor(0.0)), 
            'bound_raw': nn.Parameter(torch.tensor(10.0)),
            'noise_raw': nn.Parameter(torch.tensor(3.0)),
            't_nd_raw': nn.Parameter(torch.tensor(-2.0)),
        })

    # setting up all the parameters

    @property
    def weight_vR(self): return 8.0 * torch.sigmoid(self.params['weight_vR_raw'])
    @property
    def weight_vL(self): return 8.0 * torch.sigmoid(self.params['weight_vL_raw'])
    @property
    def weight_aR(self): return 8.0 * torch.sigmoid(self.params['weight_aR_raw'])
    @property
    def weight_aL(self): return 8.0 * torch.sigmoid(self.params['weight_aL_raw'])
    @property
    def bound(self): return .5 + 40.0 * torch.sigmoid(self.params['bound_raw'])
    @property
    def noise(self): return 2.0 + 30.0 * torch.sigmoid(self.params['noise_raw'])
    @property
    def t_nd(self): return 0.05 + 0.35 * torch.sigmoid(self.params['t_nd_raw'])    
    @property
    def gamma(self): return 0.5 + 1.5 * torch.sigmoid(self.params['gamma_raw'])

    @property
    def biasL(self): return 5.0 * torch.sigmoid(self.params['biasL_raw']) - 5.0
    @property
    def biasR(self): return 5.0 * torch.sigmoid(self.params['biasR_raw']) - 5.0
    @property
    def x0L(self): return 0.5 * torch.sigmoid(self.params['x0L_raw'])-.5
    @property
    def x0R(self): return 0.5 * torch.sigmoid(self.params['x0R_raw'])-.5

    @staticmethod
    def get_drift_rate(v, w_v, gamma, a_pro, w_a_pro, a_anti, w_a_anti, bias=0):
        return w_v * (v**gamma) + w_a_pro * a_pro - w_a_anti * a_anti + bias
    
    def simulate_cross_conditions(self, sample):
        # this I could potentailly do elsewhere... 
        aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
        vis_contrasts = np.sort(np.unique(sample.conditions['visDiff'][0]))
        aud_azimuths_t = torch.from_numpy(aud_azimuths).float().to(self.device)
        vis_contrasts_t = torch.from_numpy(vis_contrasts).float().to(self.device)

        vis_mesh, aud_mesh = torch.meshgrid(vis_contrasts_t, aud_azimuths_t, indexing='ij')
        self.original_shape = vis_mesh.shape
        vis_flat, aud_flat = vis_mesh.flatten(), aud_mesh.flatten()
        
        vR = (vis_flat > 0) * torch.abs(vis_flat)
        vL = (vis_flat < 0) * torch.abs(vis_flat)
        aR = (aud_flat > 0).float()
        aL = (aud_flat < 0).float()

        driftR = self.get_drift_rate(vR, self.weight_vR, self.gamma,
                                     aR,self.weight_aR, aL, self.weight_aL, 
                                     self.biasR)
        
        driftL = self.get_drift_rate(vL, self.weight_vL, self.gamma, 
                                     aL, self.weight_aL, aR, self.weight_aR, 
                                     self.biasL)


        # NOTE: The return values are PDFs (probability density), not probabilities.
        # The likelihood function will need to account for the time bin width (self.dt).
        pdfL_flat, pdfR_flat, pNoGo_flat = self.run_two_races(
            driftL=driftL, driftR=driftR,
            x0L=torch.full_like(driftL, -self.x0L.item()),
            x0R=torch.full_like(driftR, -self.x0R.item()),
            noise=self.noise.expand_as(driftL),
            bound=self.bound,
            t_nd=self.t_nd
        )

        pdfL_mat = pdfL_flat.view(*self.original_shape, -1)
        pdfR_mat = pdfR_flat.view(*self.original_shape, -1)
        pNoGo_mat = pNoGo_flat.view(*self.original_shape)
        
        return pdfL_mat, pdfR_mat, pNoGo_mat

    def time_binned_likelihood(self, pdf, reaction_times):
        rt_tensor = torch.as_tensor(reaction_times, device=self.device)
        if rt_tensor.numel() == 0: return 0
        rt_indices = torch.searchsorted(self.times, rt_tensor, right=True) - 1
        rt_indices = torch.clamp(rt_indices, 0, len(self.times) - 1)
        # Likelihood for a bin is PDF value * bin width
        log_likelihood = torch.log(pdf[rt_indices] * self.dt + 1e-12)
        return -torch.sum(log_likelihood)

    def tot_logLikelihood(self, sample):
        pdfL_mat, pdfR_mat, pNoGo_mat = self.simulate_cross_conditions(sample)
        
        aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
        vis_contrasts = np.sort(np.unique(sample.conditions['visDiff'][0]))
        total_nll = 0.0
        
        for i_vis, vis in enumerate(vis_contrasts):
            for i_aud, aud in enumerate(aud_azimuths):
                sub = sample.subset(audDiff=aud, visDiff=vis)
                
                nll_L = self.time_binned_likelihood(pdfL_mat[i_vis, i_aud], sub.choice_lower)
                nll_R = self.time_binned_likelihood(pdfR_mat[i_vis, i_aud], sub.choice_upper)
                
                if sub.undecided > 0:
                    nll_Nogo = -torch.log(pNoGo_mat[i_vis, i_aud] + 1e-12) * sub.undecided
                else:
                    nll_Nogo = 0
                    
                total_nll += (nll_L + nll_R + nll_Nogo)
        return total_nll
    
    # The fit function remains unchanged, as it calls the methods above.
    def fit(self, sample, num_steps=500, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print("--- Starting Race Model Fit ---")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.tot_logLikelihood(sample)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {step}: Loss is NaN or Inf. Stopping training.")
                break
            loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == num_steps - 1:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")
        
        print("--- Fit Complete ---")    


#%%
# Create an instance of the RaceModel

RaceModel = RaceModel(device='cuda')
RaceModel.plot_pdf(driftL=0, driftR=0, x0L=0, x0R=0, noise=4, bound=25, t_nd=0.2)

# %%

from utils.av_dat_manager import get_summary_dataset

# get the sample
dataset_name = 'uni_SC_nogo'
sample = get_summary_dataset(dataset_name, recompute=True, subsample=True, 
                         rt_rel_to='stim',keep_undecided = True, ctrl_only = False, max_rt = 1.5)

 #1. Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Instantiate the model on the chosen device
model = avRaceModel(device=device)

# 3. Fit the model
# The fit method now contains the optimization loop.
# It returns a dictionary of the final, optimized parameters.
final_parameters = model.fit(sample, num_steps=500, lr=0.01)


# %%


# %%
