#%%


from pathlib import Path
from utils.add_src_to_sys import *
from src.evaluate import load_evaluation,get_gain_loss_models

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\ctrl')
df = load_evaluation(path,recompute = False)
results_df = get_gain_loss_models(df)
# %%

# Split the 'stem' column into 'subjectju7u, 'hemisphere', and 'power'



# %%
plt.rcParams['font.size'] = 8

import matplotlib.pyplot as plt
import seaborn as sns

fig,ax = plt.subplots(1,1,figsize=(4,2.5))
which_metric = 'LogLik_test_norm'
sns.lineplot(data=results_df, x='param', y=which_metric, hue='type', style='stem',
              markers=True, dashes=False,legend=False,ax=ax,alpha=.3,)

sns.lineplot(data=results_df, x='param', y=which_metric, hue='type',
              markers=True, dashes=False,legend=False,ax=ax)

ax.set_ylim([-.2,1.2])
ax.axhline(1,linestyle='--',color='grey')
for label in ax.get_xticklabels():
    label.set_rotation(90)


# %%
# parameters of the full model
df_full = df[df['model_name'] == 'full']

params = results_df.param.unique()
params = [param for param in params if param not in ['ctrl']]

n_params = len(params)
fig,ax = plt.subplots(1,n_params,figsize=(n_params*2,3))
fig.subplots_adjust(wspace=1)

for i,param in enumerate(params): 
    ax[i].plot(df_full[param],'o')
    ax[i].axhline(0,linestyle='--',color='grey')
    ax[i].set_title(param)

# %%

# visualise the model

from src.my_io import read_pickle
from src.read_model import get_model


sample = path /'summary_data' / 'per_subject_Sample_train.pickle'
sample = read_pickle(sample)

model_name = 'g_vis_x0'
model = get_model(
      path = rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\ctrl\summary_data',
      sample_name='per_subject_Sample_train',
      model_name=model_name)




import plots
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.size'] = 8

aud_azimuths = [-60,-60,0,60,60]

#aud_azimuths = [0,0,0,0,0]

contrast = .5
vis_contrasts =  [-contrast,contrast,0,-contrast,contrast]

fig,ax = plt.subplots(1,len(aud_azimuths),figsize=(8,3),sharey=True, sharex=True)
data_dt = .025
scaling_factor = 3
for i,(a,v) in enumerate(zip(aud_azimuths,vis_contrasts)):
        curr_cond ={'visDiff':v,'audDiff':a}
        plots.plot_diagnostics(model=model,sample = sample, 
                        conditions=curr_cond,data_dt = data_dt,method=None,myloc=0,ax = ax[i],
                        dkwargs={'color':'purple','alpha':.6,'linewidth':1},
                        mkwargs = {'color':'purple','alpha':1,'linewidth':1.5},
                        time_on_x=True,data_plot_func='fill_between')
        
        ax[i].set_title(f'A= {a},\n V= {v}',fontsize=8)
           
ax[0].set_xlim([.1,.7])

ticks = np.array([-5,0,5])
ax[0].set_yticks(ticks)
#ax[0].set_yticklabels(np.round((ticks*data_dt),2).astype(str)) 

for axes in ax:
      axes.spines['top'].set_visible(False)
      axes.spines['right'].set_visible(False)

for axes in ax:
      axes.set_xticks([0.2,0.6])

ax[0].set_ylabel('pdf')
ax[0].set_xlabel('RT (s)')
fig.suptitle(model_name,fontsize=10,y=1.1)

# %%
