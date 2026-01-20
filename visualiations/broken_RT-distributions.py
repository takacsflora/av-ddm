

# %%

from utils.add_src_to_sys import *

import pyddm 
from utils.av_dat_manager import get_summary_dataset
from src.read_model import get_model
import visualiations.model_performace.plots as plots


dataset  = 'uni_SC_nogo'
rt_rel_to = 'stim'
sample = get_summary_dataset(dataset,recompute=True,subsample=True,rt_rel_to=rt_rel_to)

#%%
model = get_model(
      path = rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_{rt_rel_to}\summary_data',
      sample_name=dataset,
      model_name='full')

model.show()



# %%

fig = plots.av_diagnostics(sample,model = model)
#%%
import matplotlib.pyplot as plt
fig,axs = plt.subplots(1,2,figsize=(6,3),sharey=True, sharex=True)
fig1 = plots.plot_psychometric(model,sample,axctrl=axs[0],axopto=axs[1],plot_log=True)


fig,axs = plt.subplots(1,2,figsize=(6,3),sharey=True, sharex=True)
fig2 = plots.plot_chronometric(model,sample,axctrl=axs[0],axopto=axs[1],metric_type='median')
# %%


# this is way too many plots, lets try to focus on a few  values

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.size'] = 8

aud_azimuths = [-60,-60,0,60,60]

aud_azimuth_labels = [-1,-1,0,1,1]

#aud_azimuths = [0,0,0,0,0]

contrast = .5
vis_contrasts =  [-contrast,contrast,0,-contrast,contrast]

fig,ax = plt.subplots(1,len(aud_azimuths),figsize=(4,1.5),sharey=True, sharex=True)
data_dt = .025
colors = ['purple','orange']
scaling_factor = 3
for isLaser in [0]:
    for i,(a,v) in enumerate(zip(aud_azimuths,vis_contrasts)):
            curr_cond ={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
            plots.plot_diagnostics(model=model,sample = sample, 
                            conditions=curr_cond,data_dt = data_dt,method=None,myloc=0,ax = ax[i],
                            dkwargs={'color':colors[isLaser],'alpha':.6,'linewidth':1},
                            mkwargs = {'color':colors[isLaser],'alpha':1,'linewidth':1.5},
                            time_on_x=True,data_plot_func='fill_between')
            
            ax[i].set_title(f'A= {aud_azimuth_labels[i]},\n V= {v}',fontsize=8)
           
ax[0].set_xlim([.1,.7])

ticks = np.array([-5,0,5])
ax[0].set_yticks(ticks)
#ax[0].set_yticklabels(np.round((ticks*data_dt),2).astype(str)) 

for axes in ax:
      axes.spines['top'].set_visible(False)
      axes.spines['right'].set_visible(False)

for axes in ax:
      axes.set_xticks([0,0.6])
      axes.set_xticklabels([0,.6])

ax[0].set_ylabel('pdf')
ax[0].set_xlabel('RT (s)')
# %%



plt.rcParams['font.size'] = 15


#models = ['full','g_d_bx0','g_d_b','g_d_x0']
models = ['full','g_d_BL','g_d_x0','g_d_b']
n_models = len(models)
laser_to_plot = [0] + [1]*n_models
colors = ['purple' if is_laser == 0 else 'orange' for is_laser in laser_to_plot]

aud_azimuths = [-60,-60,0,60,60]

contrast = .5
vis_contrasts =  [-contrast,contrast,0,-contrast,contrast]


fig,ax = plt.subplots(n_models,len(aud_azimuths),figsize=(4,1.5*n_models),sharey=True, sharex=True)
ax = np.atleast_2d(np.array(ax))  # Ensure ax is always a 2D array
fig.patch.set_alpha(1)
for axes_row in ax:
      for axes in axes_row:
            axes.patch.set_alpha(0)
fig.subplots_adjust(hspace=-0.34,wspace=0.1)



scaling_factor = 3
laser_colors = ['purple','orange']
for model_idx,m in enumerate(models):
      model = get_model(
                  path = rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_{rt_rel_to}\summary_data',
                  sample_name=dataset,
                  model_name=m,)
      
      for isLaser in [1]:
            c = laser_colors[isLaser]
            if isLaser == 0:
                  current_sample= None 
                  mkwargs = {'color':c,'alpha':1,'linewidth':2.5,'linestyle':'--'}
            else:
                  current_sample = sample
                  mkwargs = {'color':'darkorange','alpha':1,'linewidth':2.5,'linestyle':'-'}
            for i,(a,v) in enumerate(zip(aud_azimuths,vis_contrasts)):
                        curr_cond ={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
                        plots.plot_diagnostics(model=model,sample = current_sample, 
                                    conditions=curr_cond,data_dt =.025,method=None,myloc=0,ax = ax[model_idx,i],
                                    dkwargs={'color':c,'alpha':.5,'linewidth':1},
                                    mkwargs = mkwargs,
                                    time_on_x=True,data_plot_func='fill_between')
            
            
            
ax[0,0].set_xlim([.1,.7])
#another plot where we plot several models 


for axes_row in ax:
      for axes in axes_row:
            axes.set_xticks([])
            axes.set_yticks([])
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['left'].set_visible(False)
            axes.spines['bottom'].set_visible(False)



# %%
