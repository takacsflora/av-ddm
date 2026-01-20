#%%
import numpy as np

import torch 
import torch.nn as nn
from torch.fft import rfftn, irfftn

import matplotlib.pyplot as plt
# might try to use the fftconvolution

#  implemented fast fourier convolution with torch
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

class Drift2D(nn.Module):
    def __init__(self,device = 'cuda',n_steps = 300,sample_df=None):
        super().__init__() # initialise nn.Module
        self.device = device

        # set up the 2D evidence space
        z_min, z_max, self.z_res = -5, 5, 0.05
        self.sim_zR = torch.arange(z_min, z_max, self.z_res, device=self.device)
        self.sim_zL = torch.arange(z_min, z_max, self.z_res, device=self.device)
        self.Zr, self.Zl = torch.meshgrid(self.sim_zR, self.sim_zL, indexing='ij')

        # set up the time evolution
        self.n_steps = n_steps
        self.times = torch.linspace(0, 1.5, self.n_steps, device=self.device)

        # define parameters of the model
        self.set_parameters() 

        # format the sample so that it is ready to be used for fitting
        self.prepare_sample(sample_df=sample_df)


     # set up all the parameters
    def set_parameters(self):
        self.params = nn.ParameterDict({
            'vL': nn.Parameter(torch.tensor(0.2, device=self.device)),
            'vR': nn.Parameter(torch.tensor(0.2, device=self.device)),
            'aL': nn.Parameter(torch.tensor(0.1, device=self.device)),
            'aR': nn.Parameter(torch.tensor(0.1, device=self.device)),
            'x0L': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'x0R': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'noise': nn.Parameter(torch.tensor(0.7, device=self.device)),
            't_nd': nn.Parameter(torch.tensor(.14, device=self.device)),
            'bound': nn.Parameter(torch.tensor(.9, device=self.device)),
            'gamma': nn.Parameter(torch.tensor(1.0, device=self.device)), 
        })


    def gaussian2D(self,xx=0,yy=0, sigma=0.1):
        # This function is fine, but we won't use it for the main simulation kernel anymore.
        # It's still used for the starting distribution, which is okay.
        if not isinstance(xx, torch.Tensor):
            xx = torch.tensor([xx], device=self.device)
        if not isinstance(yy, torch.Tensor):
            yy = torch.tensor([yy], device=self.device)

        MyGaussian = torch.exp(-(((self.Zr-xx.view(-1,1,1))**2) + ((self.Zl-yy.view(-1,1,1))**2)) / (2 * sigma**2))
        return MyGaussian / torch.sum(MyGaussian, dim=(-2, -1), keepdim=True) 

    def get_boundaries(self,bound=.8):

        probs = {
            'Right': torch.exp(self.Zr) / (1 + torch.exp(self.Zr) + torch.exp(self.Zl)),
            'Left': torch.exp(self.Zl) / (1 + torch.exp(self.Zr) + torch.exp(self.Zl)),
            'NoGo': 1 / (1 + torch.exp(self.Zr) + torch.exp(self.Zl)),
        }


        masks = {key: (probs[key] > bound).float() for key in probs.keys()}

        self.boundary_names = list(masks.keys())
        self.boundary_indices = {name: i for i, name in enumerate(self.boundary_names)}
        self.boundaries = torch.stack(list(masks.values()), dim=0)  # shape: (n_boundaries, Zr, Zl)
        self.n_boundaries = self.boundaries.shape[0]

        

    def make_av_conditions(self, conditions={'visDiff':[0],'audDiff':[0]}):
        # the idea is that that based on the conditions dict, you are able to make a list of conditions here
        # with the corresponding parameters (drift,x0 -- may vary based on condition)
        
        # Create meshgrid of conditions
        self.visDiff = torch.tensor(conditions['visDiff'], device=self.device,dtype=torch.float32)
        self.audDiff = torch.tensor(conditions['audDiff'], device=self.device,dtype=torch.float32)

        self.visDiff = torch.sort(self.visDiff)[0]
        self.audDiff = torch.sort(self.audDiff)[0]

        mesh_vis, mesh_aud = torch.meshgrid(self.visDiff, self.audDiff, indexing='ij')

        condition_grid = torch.stack([mesh_vis.flatten(), mesh_aud.flatten()], dim=1)

        # Calculate vR, vL, aR, aL from visDiff and audDiff
        vR = (condition_grid[:, 0] > 0).long()
        vL = (condition_grid[:, 0] < 0).long()
        aR = (condition_grid[:, 1] > 0).long()
        aL = (condition_grid[:, 1] < 0).long()

        # Concatenate new columns to condition_grid
        self.condition_combinations = torch.cat([condition_grid, 
                                                 vR.unsqueeze(1),
                                                   vL.unsqueeze(1), 
                                                   aR.unsqueeze(1),
                                                   aL.unsqueeze(1)], dim=1)



        self.condition_names = ['visDiff', 'audDiff', 'vR', 'vL', 'aR', 'aL'] 
        self.condition_indices = {name: i for i, name in enumerate(self.condition_names)}
        self.n_conditions = self.condition_combinations.shape[0]

    def prepare_sample(self, sample_df=None):
        """Prepares the sample for simulation by extracting conditions and converting to tensors.

        Args:
            sample (dict): The sample containing conditions.

        Returns:
            dict: A dictionary with conditions as tensors.
        """
        if sample_df is None:
            conditions= {'visDiff':[-1,-.5,0,.5,1],'audDiff':[-1,0,1]}
        else:
            conditions = {
                'visDiff': sample_df.visDiff.unique(),
                'audDiff':  sample_df.audDiff.unique()
            }

        self.make_av_conditions(conditions=conditions)


        # should prepare sample in exactly the same format as pdfs, so that they are ready to fit
        if sample_df is not None:         
            sample_df['choice_indices'] = sample_df['choice'].replace({-1: 2, 0: 1, 1: 0})

            ddm_conds = self.condition_combinations.cpu().numpy()

            # Efficient vectorized mapping of (visDiff, audDiff) to condition_indices
            cond_map = {(v, a): i for i, (v, a) in enumerate(ddm_conds[:,[self.condition_indices['visDiff'], self.condition_indices['audDiff']]])}
            sample_df['condition_indices'] = [
                cond_map.get((row['visDiff'], row['audDiff']), -1) for _, row in sample_df.iterrows()
            ]

            # now we will create the approporiate tensors
            self.RTs = torch.tensor(sample_df['RT'].values, device=self.device, dtype=torch.float32) # the RT 
            rt_indices = torch.searchsorted(self.times, self.RTs, right=True) - 1
            self.rt_indices = torch.clamp(rt_indices, 0, len(self.times) - 1) # time index that the RT belongs to 
            self.choice_indices = torch.tensor(sample_df['choice_indices'].values, device=self.device, dtype=torch.int64) # the choice index
            self.condcomb_indices = torch.tensor(sample_df['condition_indices'].values, device=self.device, dtype=torch.int64) # which condition it belongs to in condition combinations

    @staticmethod
    def drift_function(vis, weight_vis, aud_pro, weight_aud_pro,aud_anti,weight_aud_anti, gamma):        
        return weight_vis * (vis**gamma) + weight_aud_pro * (aud_pro) - weight_aud_anti * (aud_anti) 

    @staticmethod
    def x0_function(x0):
        return x0
    
    def get_params_per_condition(self):
        

        return {
            'driftR': self.drift_function(self.condition_combinations[:,self.condition_indices['vR']], self.params['vR'], 
                                          self.condition_combinations[:,self.condition_indices['aR']], self.params['aR'], 
                                          self.condition_combinations[:,self.condition_indices['aL']], self.params['aL'], 
                                          self.params['gamma']),
            'driftL': self.drift_function(self.condition_combinations[:,self.condition_indices['vL']], self.params['vL'],
                                          self.condition_combinations[:,self.condition_indices['aL']], self.params['aL'], 
                                          self.condition_combinations[:,self.condition_indices['aR']], self.params['aR'], 
                                          self.params['gamma']),
            'x0R': self.x0_function(self.params['x0R']),
            'x0L': self.x0_function(self.params['x0L']),
        }
    
    def apply_nondecision_time(self, pdfs):
        """Applies non-decision time to the probability density functions.

        Args:
            pdfs (torch.tensor): The probability density functions.
            t_nd (float, optional): Non-decision time. Defaults to None.

        Returns:
            torch.tensor: Probability density functions with non-decision time applied.
        """
        # Apply non-decision time by shifting the pdfs
        t_res = self.times[1] - self.times[0]  # time resolution
        pdfs = torch.roll(pdfs, shifts=int(self.params['t_nd'] / t_res), dims=0)

        return pdfs

    def simulate(self):

        # This function is not used in the main simulation kernel anymore.
        # It was replaced by a more efficient method using FFT convolution.

        # set up the boundaries
        self.get_boundaries(bound=self.params['bound']) 
        boundaries = self.boundaries
        combined_boundaries = torch.sum(boundaries, dim=0)  # sum over boundaries to get the combined boundary mask
        
        # set up the convolutional kernels
        params = self.get_params_per_condition() # parameter matrix by conditions

        update_kernel = self.gaussian2D(xx=params['driftR'],yy=params['driftL'],sigma=self.params['noise']) # conditions x Zr x Zl

        # initialise the collected probability density
        p_density_history = torch.zeros((self.n_steps, self.n_conditions,self.Zr.shape[0], self.Zr.shape[1]), device=self.device) # time x Zr x Zl x conditions

        pdfs = torch.zeros((self.n_steps,self.n_conditions, self.n_boundaries),device=self.device) # time x conditions x boundaries

        p_density_history[0, :, :, :] = self.gaussian2D(xx=params['x0R'], yy=params['x0L'], sigma=self.z_res) 

        for nt in range(1, self.n_steps):
            current_density = fftconvolve_torch(p_density_history[nt-1, :, :, :], update_kernel,self.device)  # convolve the previous density with the update kernel
            
            # Apply the boundaries to the probability density
            pdfs[nt] = torch.sum(current_density.unsqueeze(1) * boundaries.unsqueeze(0), dim=(-2, -1))

            # absorb the committed choices
            current_density = current_density * (1 - combined_boundaries)
            p_density_history[nt, :, :, :] = current_density

        
        # add nondecision time
        pdfs = self.apply_nondecision_time(pdfs)

        return pdfs, p_density_history
    
    # couple of visualizations to check the results
    @staticmethod
    def visualize_drift_process(p_density_history):
        """Visualizes the drift process over time, for the 0th condition

        Args:
            p_density_history (torch.tensor): The probability density history over time. 
        """
        p_density_history = p_density_history.cpu().numpy()
        n_steps = p_density_history.shape[0]
        _,ax = plt.subplots(1,n_steps, figsize=(15, 5), sharex=True, sharey=True)

        for i in range(n_steps):
            ax[i].matshow(p_density_history[i,0,:,:], aspect='auto', origin='lower',cmap='Greys')
            ax[i].set_title(f'Step {i+1}')
            ax[i].axis('off')

        plt.tight_layout()

    def reshape_conditions(self,pdfs):
        """_summary_

        Args:
            pdfs (times x conditions x boundaries): generated pdfs 

        Returns:
            pdfs (times x audDiff x visDiff  x boundaries): reshaped pdfs
        """


        visDiff_per_cond = (self.condition_combinations[:,self.condition_indices['visDiff']])
        audDiff_per_cond = (self.condition_combinations[:,self.condition_indices['audDiff']])

        unique_vis = self.visDiff
        unique_aud = self.audDiff

        n_vis = len(unique_vis)
        n_aud = len(unique_aud)
        n_choices = pdfs.shape[2]
        n_steps = pdfs.shape[0]

        reshaped_pdfs = torch.zeros((n_steps, n_aud, n_vis, n_choices), device=pdfs.device)

        for i, v in enumerate(unique_vis):
            for j, a in enumerate(unique_aud):
                cond_mask = (visDiff_per_cond == v) & (audDiff_per_cond == a)
                reshaped_pdfs[:, j, i, :] = pdfs[:, cond_mask, :].squeeze(1)
        # Set the boundary indices for Right, Left, and NoGo
        # mmh maybe actually move this into the  prepare sample function
        if hasattr(self, 'condcomb_indices') and hasattr(self, 'choice_indices'):
            # Create a tensor to store empirical probabilities: (n_aud, n_vis, n_choices)

            # Calculate empirical pdfs for each timepoint and choice
            empirical_pdfs = torch.zeros((n_steps, n_aud, n_vis, n_choices), device=pdfs.device)
            n_trials = self.condcomb_indices.shape[0]
            for i, v in enumerate(unique_vis):
                for j, a in enumerate(unique_aud):
                    cond_mask = (visDiff_per_cond == v) & (audDiff_per_cond == a)
                    cond_idx = torch.where(cond_mask)[0][0]
                    for k in range(n_choices):
                        # For each timepoint, count samples with matching RT index, condition, and choice
                        for t in range(n_steps):
                            sample_mask = (
                                (self.condcomb_indices == cond_idx) &
                                (self.choice_indices == k) &
                                (self.rt_indices == t)
                            )
                            empirical_pdfs[t, j, i, k] = sample_mask.sum() 

            n_trials_per_cond = empirical_pdfs.sum(dim=0).sum(dim=-1)
            # Stack n_trials_per_cond to match empirical_pdfs shape for division
            n_trials_per_cond_stacked = n_trials_per_cond.unsqueeze(0).unsqueeze(-1).expand_as(empirical_pdfs)
            empirical_pdfs = empirical_pdfs / n_trials_per_cond_stacked
            self.empirical_pdfs = empirical_pdfs 
            self.empirical_probs = torch.sum(self.empirical_pdfs,dim=0)

        
        return reshaped_pdfs

    def visualize_pdfs(self,pdfs,**pkws):
        # to rewrite with the new reshape function
        
        pdfs_ = self.reshape_conditions(pdfs).cpu().numpy()
        times = self.times.cpu().numpy()

        if hasattr(self, 'empirical_pdfs'):
            empirical_pdfs_ = self.empirical_pdfs.cpu().numpy()
        
        n_aud, n_vis = pdfs_.shape[1:3]

        fig, ax = plt.subplots(n_aud,n_vis, figsize=(15, 10), sharex=True, sharey=True)

        for idx_a in range(n_aud):
            for idx_v in range(n_vis):
                # find the index of the condition
      
                # get the pdfs for this condition
                current_pdfs = pdfs_[:, idx_a,idx_v, :]

                # plot the pdfs of this condition
                ax[n_aud-idx_a-1, idx_v].plot(times, current_pdfs[:,self.boundary_indices['Right']],**pkws)
                ax[n_aud-idx_a-1, idx_v].plot(times, -current_pdfs[:,self.boundary_indices['Left']],**pkws)

                if hasattr(self, 'empirical_pdfs'):
                    current_data = empirical_pdfs_[:, idx_a,idx_v, :]
                    ax[n_aud-idx_a-1, idx_v].fill_between(times, current_data[:,self.boundary_indices['Right']],**pkws)
                    ax[n_aud-idx_a-1, idx_v].fill_between(times, -current_data[:,self.boundary_indices['Left']],**pkws)


        for a in ax.flatten():
            a.axhline(0, color='black', lw=1, ls='--')
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

    def plot_probabilities(self,pdfs, numerator='Right', denominator='Left',ax=None):
        """Visualizes the probabilities of the specified numerator and denominator conditions.

        Args:
            pdfs (torch.tensor): The probability density functions.
            numerator (str): The condition to use as the numerator.
            denominator (str): The condition to use as the denominator.
        """
        pdfs_ = self.reshape_conditions(pdfs)

        probabilities = torch.sum(pdfs_, dim=0)

        if denominator is None:
            plot_probs = probabilities[:, :, self.boundary_indices[numerator]]
        else:
            plot_probs = torch.log(probabilities[:, :, self.boundary_indices[numerator]] / probabilities[:, :, self.boundary_indices[denominator]])

        if hasattr(self, 'empirical_probs'):
            if denominator is None:
                empirical_probs = self.empirical_probs[:, :, self.boundary_indices[numerator]]
            else:
                empirical_probs = torch.log(self.empirical_probs[:, :, self.boundary_indices[numerator]] / self.empirical_probs[:, :, self.boundary_indices[denominator]])
            
        plot_probs = plot_probs.cpu().numpy()
        if hasattr(self, 'empirical_probs'):
            empirical_probs = empirical_probs.cpu().numpy()

        visDiffs = self.visDiff.cpu().numpy()
        colors = plt.cm.coolwarm(np.linspace(0, 1, plot_probs.shape[0]))
        
        if ax is None: 
            _,ax =plt.subplots(figsize=(5, 5))
        for i in range(plot_probs.shape[0]):
            ax.plot(visDiffs, plot_probs[i], color=colors[i])

            if hasattr(self, 'empirical_probs'):
                ax.plot(visDiffs, empirical_probs[i], color=colors[i], marker='o',linestyle='None')

    def tot_likelihood(self):
        pdfs, _ = self.simulate()
        
        continuous_ll_sample_indices = self.choice_indices<2

        # Compute log-likelihood for each sample (excluding NoGo/undecided trials)
        log_likelihoods = torch.log(
            torch.clamp(
                pdfs[
                    self.rt_indices[continuous_ll_sample_indices],
                    self.condcomb_indices[continuous_ll_sample_indices],
                    self.choice_indices[continuous_ll_sample_indices]
                ],
                min=1e-10
            )
        )
        tot_ll_continuous = log_likelihoods.sum()


        # now calculate the nogos ll 
        nogo_indices = self.choice_indices==2
        prob_per_cond_nogo = torch.sum(pdfs[:,:,2], dim=0)

        # For NoGo trials, sum the probability of NoGo across all time steps for each condition
        nogo_ll = torch.log(torch.clamp(prob_per_cond_nogo[self.condcomb_indices[nogo_indices]], min=1e-10)).sum()
        tot_ll = tot_ll_continuous + nogo_ll
        return -tot_ll

    def fit(self,num_epochs=1000, lr=0.01, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.tot_likelihood()
            loss.backward()
            optimizer.step()
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

    @torch.no_grad()
    def predict(self):
        """Predicts the probability density functions for the currently set parameters."""
        self.eval()

        pdfs, p_density_history = self.simulate()

        return pdfs, p_density_history

#%%
from utils.av_dat_manager import get_summary_dataset

# get the sample
dataset_name = 'uni_SC_nogo'
sample_df,_ = get_summary_dataset(dataset_name, recompute=True, subsample=True, return_df=True,
                         rt_rel_to='stim',keep_undecided = True, ctrl_only = True, max_rt = 1.5)

# example simulation
ddm = Drift2D(device='cuda',n_steps = 50,sample_df=sample_df)

ddm.fit(num_epochs=200, lr=0.01, verbose=True)

#%%

# I am going to do a race. For this I will do 

BoundL = 4
boundR = 3
zzzL = ddm.Zl.cpu().numpy()>BoundL
zzzR = ddm.Zr.cpu().numpy()>boundR

fig,ax = plt.subplots(1,1,figsize=(5,5),sharey=True, sharex=True)
ax.matshow(zzzL, cmap='Greys', vmin=0, vmax=1,alpha=.5)
ax.matshow(zzzR, cmap='Greys', vmin=0, vmax=1,alpha=.5)
ax.invert_yaxis()
#%%
pdfs,p_density_history = ddm.predict() 

#ddm.visualize_drift_process(p_density_history)


model_params = {k:ddm.params[k].item() for k in ddm.params.keys()}
# %%# %%
ddm.visualize_pdfs(pdfs, color='black', alpha=.5)



# %%
ddm.plot_probabilities(pdfs, numerator='NoGo', denominator=None)


# %%
ddm.plot_probabilities(pdfs, numerator='Right', denominator='Left')

# %%
ddm.plot_probabilities(pdfs, numerator='Right', denominator='NoGo')
