#%%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.fft import rfftn, irfftn
import numpy as np

# (fftconvolve_torch remains unchanged)
def fftconvolve_torch(in1, in2, device):
    # This function is correct and handles batch convolutions, which is what we need.
    s1 = torch.tensor(in1.shape[-2:], device=device)
    s2 = torch.tensor(in2.shape[-2:], device=device)
    shape_tensor = s1 + s2 - 1
    shape_tuple = tuple(shape_tensor.tolist())
    
    # The fft functions work on the last N dimensions, so this will
    # correctly perform a batch of 2D convolutions if in1 and in2 have
    # leading batch dimensions (e.g., shape [n_vis, n_aud, z_dim, z_dim]).
    sp1 = rfftn(in1, s=shape_tuple, dim=(-2, -1))
    sp2 = rfftn(in2, s=shape_tuple, dim=(-2, -1))
    
    ret = irfftn(sp1 * sp2, s=shape_tuple, dim=(-2, -1))

    start_indices = (shape_tensor - s1) // 2
    # This slicing correctly crops the result for 'same' mode convolution.
    slices = [slice(None)] * (len(ret.shape) - 2) + \
             [slice(int(start_indices[0]), int(start_indices[0] + s1[0])),
              slice(int(start_indices[1]), int(start_indices[1] + s1[1]))]
    return ret[tuple(slices)]

class ThreeChoiceDecisionTorch(nn.Module):
    # This class is correct and stable
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        z_min, z_max, self.z_res = -5, 5, 0.05
        self.sim_zR = torch.arange(z_min, z_max, self.z_res, device=self.device)
        self.sim_zL = torch.arange(z_min, z_max, self.z_res, device=self.device)
        self.Zr, self.Zl = torch.meshgrid(self.sim_zR, self.sim_zL, indexing='ij')
        self.prob_keys = ['Right', 'Left', 'NoGo']
        self.probs = {
            'Right': torch.exp(self.Zr) / (1 + torch.exp(self.Zr) + torch.exp(self.Zl)),
            'Left': torch.exp(self.Zl) / (1 + torch.exp(self.Zr) + torch.exp(self.Zl)),
            'NoGo': 1 / (1 + torch.exp(self.Zr) + torch.exp(self.Zl)),
        }
        self.n_steps = 100
        self.times = torch.linspace(0, 1.5, self.n_steps, device=self.device)

    def get_gaussian(self, sigma=0.1):
        # This function is fine, but we won't use it for the main simulation kernel anymore.
        # It's still used for the starting distribution, which is okay.
        MyGaussian = torch.exp(-(((self.Zr)**2) + ((self.Zl)**2)) / (2 * sigma**2))
        return MyGaussian / torch.sum(MyGaussian, dim=(-2, -1), keepdim=True)
    
    def apply_nondectime(self, probabilities, t_nd):
        shift_seconds = t_nd.item() if isinstance(t_nd, torch.Tensor) else t_nd
        dt = (self.times[1] - self.times[0]).item()
        shift_steps = int(round(shift_seconds / dt))
        ps_shifted = torch.zeros_like(probabilities)
        if shift_steps > 0 and shift_steps < probabilities.shape[-1]:
            ps_shifted[..., shift_steps:] = probabilities[..., :-shift_steps]
        elif shift_steps <= 0:
            ps_shifted = probabilities
        return torch.clamp(ps_shifted, 1e-10)

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


class AVDecisionTorch(ThreeChoiceDecisionTorch):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.params = nn.ParameterDict({
            'weight_vR_raw': nn.Parameter(torch.tensor(0.0)),
            'weight_vL_raw': nn.Parameter(torch.tensor(0.0)),
            'weight_aR': nn.Parameter(torch.tensor(0.3)), 
            'weight_aL': nn.Parameter(torch.tensor(0.3)),
            'x0R': nn.Parameter(torch.tensor(0.0)), 'x0L': nn.Parameter(torch.tensor(0.0)),
            'gamma_raw': nn.Parameter(torch.tensor(0.0)), 
            'bound_raw': nn.Parameter(torch.tensor(1.7)), 
            'sigma_raw': nn.Parameter(torch.tensor(-0.22)),
            't_nd_raw': nn.Parameter(torch.tensor(-2.0)),
        })
        
    @property
    def weight_vR(self): return -4 + 4 * torch.sigmoid(self.params['weight_vR_raw'])
    @property
    def weight_vL(self): return -4 + 4 * torch.sigmoid(self.params['weight_vL_raw'])
    @property
    def bound(self): return 0.5 + 0.49 * torch.sigmoid(self.params['bound_raw'])
    @property
    def sigma(self): return 0.05 + 10 * torch.sigmoid(self.params['sigma_raw'])
    @property
    def t_nd(self): return 0.03 + 0.27 * torch.sigmoid(self.params['t_nd_raw'])    
    @property
    def gamma(self): return 0.5 + 0.5 * torch.sigmoid(self.params['gamma_raw'])

    @staticmethod
    def get_starting_position(a_pro, w_a_pro, a_anti, w_a_anti, x0):
        return w_a_pro * a_pro - w_a_anti * a_anti + x0
    
    @staticmethod 
    def get_drift_rate(v,w_v,gamma):
        # Calculate the drift based on visual and auditory inputs
        return w_v * (v**gamma)

    def simulate_cross_conditions(self, sample):
        aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
        vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
        aud_azimuths_t = torch.from_numpy(aud_azimuths).float().to(self.device)
        vis_contrasts_t = torch.from_numpy(vis_contrasts).float().to(self.device)

        masks = {key: (self.probs[key] > self.bound).float() for key in self.prob_keys}
        combined_mask = torch.sum(torch.stack(list(masks.values())), dim=0)

        # Correctly get vis_mesh and aud_mesh from unique values
        vis_mesh, aud_mesh = torch.meshgrid(vis_contrasts_t, aud_azimuths_t, indexing='ij')
        
        vR = (vis_mesh > 0) * torch.abs(vis_mesh)
        vL = (vis_mesh < 0) * torch.abs(vis_mesh)
        # Transpose aud_mesh to match vis_mesh's condition mapping
        aR = (aud_mesh > 0).float()
        aL = (aud_mesh < 0).float()

        p = self.params
        start_R = self.get_starting_position(aR, p['weight_aR'], aL, p['weight_aL'], p['x0R'])
        start_L = self.get_starting_position(aL, p['weight_aL'], aR, p['weight_aR'], p['x0L'])

        start_matrices = torch.exp(-((self.Zr[None, None, :, :] - start_R[:, :, None, None])**2 +
                                     (self.Zl[None, None, :, :] - start_L[:, :, None, None])**2) / (2 * 0.1**2))
        
        start_matrices = start_matrices / torch.sum(start_matrices, dim=(-2, -1), keepdims=True)

        # --- NEW CORRECTED LOGIC FOR STIMULUS-DEPENDENT KERNEL ---

        # 1. Calculate drift rate (evidence units per second) for each condition
        drift_rate_R = self.get_drift_rate(vR, self.weight_vR, self.gamma)
        drift_rate_L = self.get_drift_rate(vL, self.weight_vL, self.gamma)
        
        # 2. Calculate drift amount for a single timestep (dt)
        dt = (self.times[1] - self.times[0]).item()
        drift_per_step_R = drift_rate_R * dt
        drift_per_step_L = drift_rate_L * dt

        # 3. Create the batch of drift-diffusion kernels using broadcasting
        # The mean of each Gaussian is the drift for that condition.
        # The standard deviation is self.sigma (diffusion per step).
        kernel_variance = self.sigma**2

        # Expand dimensions for broadcasting:
        # drift_R/L shape: (n_vis, n_aud) -> (n_vis, n_aud, 1, 1)
        # Zr/Zl shape: (Z_dim_r, Z_dim_l) -> (1, 1, Z_dim_r, Z_dim_l)
        drift_kernels = torch.exp(
            -(( (self.Zr[None, None, :, :] - drift_per_step_R[:, :, None, None])**2 + 
                (self.Zl[None, None, :, :] - drift_per_step_L[:, :, None, None])**2 ) 
              / (2 * kernel_variance))
        )
        
        # 4. Normalize each kernel in the batch individually
        drift_kernels = drift_kernels / torch.sum(drift_kernels, dim=(-2, -1), keepdims=True)
        # drift_kernels now has shape [n_vis, n_aud, Z_dim_r, Z_dim_l]
        # and is ready for batch convolution.

        # --- END OF MODIFIED LOGIC ---

        probDensMovie_t = start_matrices
        results_list = [] 

        for nt in range(self.n_steps):
            if nt > 0:
                # Convolve with the stimulus-dependent batch of kernels
                probDensMovie_t = fftconvolve_torch(probDensMovie_t, drift_kernels, self.device)

            current_step_probs_by_choice = {
                key: torch.sum(probDensMovie_t * masks[key], dim=(-2, -1)) for key in self.prob_keys
            }
            
            # This is the probability of crossing each bound *at this specific timestep*
            # It's the "probability mass flux" across the boundary.
            dt_prob = torch.stack([current_step_probs_by_choice[key] for key in self.prob_keys], dim=-1)
            results_list.append(dt_prob)
            
            # Update the probability density (remove absorbed mass)
            probDensMovie_t = probDensMovie_t * (1 - combined_mask)
        
        # Stack into (n_vis, n_aud, n_steps, n_choices)
        results = torch.stack(results_list, dim=2)
        
        # Normalize PDFs to 1 for each condition
        # Sum of probabilities over time should be <= 1.
        total_prob_absorbed = torch.sum(results, dim=2, keepdim=True)
        # Avoid division by zero if a condition has zero probability of ever making a choice
        total_prob_absorbed = torch.clamp(total_prob_absorbed, 1e-10) 
        
        # `results` now contains the probability density function of reaction times (RT-PDF)
        rt_pdfs = results / dt

        # Apply non-decision time
        # Permute to (batch_dims..., n_choices, time)
        rt_pdfs_permuted = rt_pdfs.permute(0, 1, 3, 2)
        shifted_pdfs = self.apply_nondectime(rt_pdfs_permuted, self.t_nd)
        
        # The output should be probabilities for each time bin, not a PDF.
        # So we multiply by dt again.
        final_probs = shifted_pdfs.permute(0, 1, 3, 2) * dt
            
        return final_probs

    def tot_logLikelihood(self, sample):
        probabilities = self.simulate_cross_conditions(sample)
        aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
        vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
        total_log_likelihood = 0.0
        
        for i_vis, vis in enumerate(vis_contrasts):
            for i_aud, aud in enumerate(aud_azimuths):
                sub = sample.subset(audDiff=aud, visDiff=vis)
                # Probabilities for this specific condition (vis, aud)
                # Shape: [n_steps, n_choices]
                current_probs = probabilities[i_vis, i_aud] 
                
                # Likelihood for Right, Left choices based on their RTs
                # Assuming choice_upper is Right (index 0), choice_lower is Left (index 1)
                total_log_likelihood += self.time_binned_likelihood(current_probs[:, 0], sub.choice_upper)
                total_log_likelihood += self.time_binned_likelihood(current_probs[:, 1], sub.choice_lower)
                
                # Likelihood for NoGo choices
                # The probability of a NoGo is 1 minus the total probability of making a choice
                prob_go_total = torch.sum(current_probs[:, :2])
                prob_nogo_final = 1.0 - prob_go_total
                
                if sub.undecided > 0:
                    total_log_likelihood += -torch.log(torch.clamp(prob_nogo_final, 1e-10)) * sub.undecided
                    
        return total_log_likelihood

    def fit(self, sample, num_steps=500, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        print("Starting training...")
        print(f"Device: {self.device}")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.tot_logLikelihood(sample)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN or Inf. Stopping training.")
                # You might want to print parameter values here for debugging
                for name, param in self.named_parameters():
                    print(f"{name}: value={param.data.item()}, grad={param.grad}")
                break
                
            loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == num_steps - 1:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")
                print(f"  bound: {self.bound.item():.3f}, sigma: {self.sigma.item():.3f}, t_nd: {self.t_nd.item():.3f}, gamma: {self.gamma.item():.3f}")
                print(f"  w_vR: {self.weight_vR.item():.3f}, w_vL: {self.weight_vL.item():.3f}")
        
        return {
            'weight_vR': self.weight_vR.item(), 'weight_vL': self.weight_vL.item(),
            'weight_aR': self.params['weight_aR'].item(), 'weight_aL': self.params['weight_aL'].item(),
            'x0R': self.params['x0R'].item(), 'x0L': self.params['x0L'].item(),
            'gamma': self.gamma.item(), 'bound': self.bound.item(),
            'sigma': self.sigma.item(), 't_nd': self.t_nd.item()
        }
        
    @torch.no_grad() # Crucial decorator for inference
    def predict(self, params_dict, sample):
        """
        Runs a simulation with a given set of parameters and returns the results
        as NumPy arrays for plotting or analysis. This is the inference-mode
        equivalent of simulate_cross_conditions.
        """
        # Set the model to evaluation mode (disables things like dropout if you had any)
        self.eval()

        # --- Setup conditions (same as in simulation) ---
        aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
        vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
        aud_azimuths_t = torch.from_numpy(aud_azimuths).float().to(self.device)
        vis_contrasts_t = torch.from_numpy(vis_contrasts).float().to(self.device)

        vis_mesh, aud_mesh = torch.meshgrid(vis_contrasts_t, aud_azimuths_t, indexing='ij')
        
        vR = (vis_mesh > 0) * torch.abs(vis_mesh)
        vL = (vis_mesh < 0) * torch.abs(vis_mesh)
        aR = (aud_mesh > 0).float()
        aL = (aud_mesh < 0).float()
        
        # --- Re-build the simulation using the provided parameters from the dictionary ---
        p = params_dict # Use the provided dictionary for parameters
        
        # Use the parameter values from the dictionary
        masks = {key: (self.probs[key] > p['bound']).float() for key in self.prob_keys}
        combined_mask = torch.sum(torch.stack(list(masks.values())), dim=0)

        start_R = self.get_starting_position(aR, p['weight_aR'], aL, p['weight_aL'], p['x0R'])
        start_L = self.get_starting_position(aL, p['weight_aL'], aR, p['weight_aR'], p['x0L'])

        start_matrices = torch.exp(-((self.Zr[None, None, :, :] - start_R[:, :, None, None])**2 +
                                    (self.Zl[None, None, :, :] - start_L[:, :, None, None])**2) / (2 * 0.1**2))
        start_matrices = start_matrices / torch.sum(start_matrices, dim=(-2, -1), keepdims=True)

        # --- Create the stimulus-dependent kernel using parameters from the dictionary ---
        drift_rate_R = self.get_drift_rate(vR, p['weight_vR'], p['gamma'])
        drift_rate_L = self.get_drift_rate(vL, p['weight_vL'], p['gamma'])
        
        dt = (self.times[1] - self.times[0]).item()
        drift_per_step_R = drift_rate_R * dt
        drift_per_step_L = drift_rate_L * dt

        kernel_variance = p['sigma']**2

        drift_kernels = torch.exp(
            -(( (self.Zr[None, None, :, :] - drift_per_step_R[:, :, None, None])**2 + 
                (self.Zl[None, None, :, :] - drift_per_step_L[:, :, None, None])**2 ) 
            / (2 * kernel_variance))
        )
        drift_kernels = drift_kernels / torch.sum(drift_kernels, dim=(-2, -1), keepdims=True)

        # --- Run the simulation loop (same logic as fit) ---
        probDensMovie_t = start_matrices
        results_list = [] 

        for nt in range(self.n_steps):
            if nt > 0:
                probDensMovie_t = fftconvolve_torch(probDensMovie_t, drift_kernels, self.device)

            current_step_probs_by_choice = {
                key: torch.sum(probDensMovie_t * masks[key], dim=(-2, -1)) for key in self.prob_keys
            }
            dt_prob = torch.stack([current_step_probs_by_choice[key] for key in self.prob_keys], dim=-1)
            results_list.append(dt_prob)
            probDensMovie_t = probDensMovie_t * (1 - combined_mask)
        
        results = torch.stack(results_list, dim=2)
        
        # --- Final processing using parameters from the dictionary ---
        rt_pdfs = results / dt
        rt_pdfs_permuted = rt_pdfs.permute(0, 1, 3, 2)
        # Use t_nd from the parameter dictionary
        shifted_pdfs = self.apply_nondectime(rt_pdfs_permuted, p['t_nd'])
        final_probs = shifted_pdfs.permute(0, 1, 3, 2) * dt
            
        # --- CONVERT TO NUMPY FOR PLOTTING ---
        # .detach() severs the tensor from the computation graph.
        # .cpu() moves the tensor to the CPU (if it was on the GPU).
        # .numpy() converts the CPU tensor to a NumPy array.
        return final_probs.detach().cpu().numpy()
    

# %%
from utils.av_dat_manager import get_summary_dataset

# get the sample
dataset_name = 'uni_SC_nogo'
sample = get_summary_dataset(dataset_name, recompute=True, subsample=True, 
                         rt_rel_to='stim',keep_undecided = True, ctrl_only = False, max_rt = 1.5)

 #1. Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Instantiate the model on the chosen device
model = AVDecisionTorch(device=device)

# 3. Fit the model
# The fit method now contains the optimization loop.
# It returns a dictionary of the final, optimized parameters.
final_parameters = model.fit(sample, num_steps=500, lr=0.01)

print("\nOptimized Parameters:")
for key, value in final_parameters.items():
    print(f"  {key}: {value:.4f}")

# %%
import matplotlib.pyplot as plt
import numpy as np

# (Assume your model has been fit and you have `final_parameters` and `sample`)

# 1. Generate predictions using the fitted parameters
print("Generating predictions with final parameters...")
predictions_np = model.predict(final_parameters, sample)

# 2. Get the time axis and condition info
times_np = model.times.cpu().numpy()
aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
n_aud = len(aud_azimuths)
n_vis = len(vis_contrasts)
bin_width = times_np[1] - times_np[0]

# 3. Plot both model predictions and data histograms
print("Plotting results...")
fig, axs = plt.subplots(n_aud, n_vis, figsize=(18, 10), sharex=True, sharey=True)
fig.suptitle('Model Predictions vs. Data Histograms', fontsize=16)

for i_aud, aud in enumerate(aud_azimuths):
    for i_vis, vis in enumerate(vis_contrasts):
        ax = axs[n_aud - 1 - i_aud, i_vis] # Flips y-axis for intuitive layout

        # --- a) Plot Model Predictions (as lines) ---
        # FIX: Indexing order changed to [vis, aud] to match predictions_np shape
        ax.plot(times_np, predictions_np[i_vis, i_aud, :, 0] / bin_width, 
                label='Model Right', color='crimson', linewidth=2)
        ax.plot(times_np, -predictions_np[i_vis, i_aud, :, 1] / bin_width, 
                label='Model Left', color='royalblue', linewidth=2)

        # --- b) Get and Plot Data Histograms (as filled steps) ---
        sample_subset = sample.subset(audDiff=aud, visDiff=vis)
        rt_right = sample_subset.choice_upper
        rt_left = sample_subset.choice_lower
        n_undecided = sample_subset.undecided
        n_trials = len(rt_right) + len(rt_left) + n_undecided

        if n_trials > 0:
            bins = np.append(times_np, times_np[-1] + bin_width) # Ensure bins cover full range
            hist_right, bin_edges = np.histogram(rt_right, bins=bins)
            hist_left, _ = np.histogram(rt_left, bins=bins)
            
            # Normalize to probability density to match model plot
            norm_hist_right = hist_right / (n_trials * bin_width)
            norm_hist_left = hist_left / (n_trials * bin_width)

            ax.step(bin_edges[:-1], norm_hist_right, 
                    where='post', color='lightcoral', alpha=0.8, label='Data Right')
            ax.step(bin_edges[:-1], -norm_hist_left, 
                    where='post', color='lightskyblue', alpha=0.8, label='Data Left')

        # --- c) Formatting ---
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
        ax.grid(True, linestyle=':', alpha=0.5)

        if i_aud == n_aud - 1: ax.set_xlabel('Time (s)')
        if i_vis == 0: ax.set_ylabel(f'Aud = {aud}')
        if i_aud == 0: ax.set_title(f'Vis = {vis:.2f}')

fig.text(0.06, 0.5, 'Choice Probability Density', ha='center', va='center', rotation='vertical')
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
plt.show()
# %%

#%%

# plot the psychometric function
import matplotlib.pyplot as plt
import seaborn as sns

# (Assumes predictions_np is already generated from the previous step)
# predictions_np = model.predict(final_parameters, sample)

# Sum probabilities over the time dimension to get total choice probability
# predictions_np has shape [n_vis, n_aud, n_steps, n_choices]
probs_model = predictions_np.sum(axis=2)  # Shape becomes [n_vis, n_aud, n_choices]

# Calculate probability of NoGo for the model
prob_go_model = probs_model[:, :, 0] + probs_model[:, :, 1]
prob_nogo_model = 1.0 - prob_go_model

# Compute empirical probabilities from the sample
empirical_probs = np.zeros((n_vis, n_aud, 3))  # Use [vis, aud] order

for i_vis, vis in enumerate(vis_contrasts):
    for i_aud, aud in enumerate(aud_azimuths):
        subset = sample.subset(audDiff=aud, visDiff=vis)
        n_right = len(subset.choice_upper)
        n_left = len(subset.choice_lower)
        n_nogo = subset.undecided
        n_total = n_right + n_left + n_nogo
        if n_total > 0:
            empirical_probs[i_vis, i_aud, 0] = n_right / n_total  # Right
            empirical_probs[i_vis, i_aud, 1] = n_left / n_total   # Left
            empirical_probs[i_vis, i_aud, 2] = n_nogo / n_total    # NoGo
        else:
            empirical_probs[i_vis, i_aud, :] = np.nan

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, sharex=True)
colors = sns.color_palette("coolwarm", n_colors=n_aud)

# Plot for each auditory condition
for i_aud, aud in enumerate(aud_azimuths):
    aud_label = f'Aud={aud}'
    # Plot Right Choices
    axs[0].plot(vis_contrasts, empirical_probs[:, i_aud, 0], 'o', color=colors[i_aud], alpha=0.7)
    axs[0].plot(vis_contrasts, probs_model[:, i_aud, 0], '-', color=colors[i_aud], label=aud_label)

    # Plot No-Go Choices
    axs[1].plot(vis_contrasts, empirical_probs[:, i_aud, 2], 'o', color=colors[i_aud], alpha=0.7)
    axs[1].plot(vis_contrasts, prob_nogo_model[:, i_aud], '-', color=colors[i_aud])

axs[0].set_title('Right Choices')
axs[1].set_title('No-Go Choices')
axs[0].set_xlabel('Visual Contrast')
axs[1].set_xlabel('Visual Contrast')
axs[0].set_ylabel('Proportion of Choices')
fig.legend(loc='upper right', bbox_to_anchor=(1.15, 0.9))
plt.tight_layout()
plt.show()
# %%
