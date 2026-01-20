#%%



# I don't know what is happening with the batch fit so I am need to look at the individual fits

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from utils.add_src_to_sys import *

from src.my_io import read_pickle
from src.evaluate import get_dataset

import plots

SAVE_PATH = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\DDM'

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})



#%% load data 

path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\all_mice\with_undecided\ctrl')
dataset = get_dataset(path)

dataset = dataset[dataset.stem=='summary_sample'].iloc[0]

sample = read_pickle(dataset.train_path)


#%% first a  single diagnostic plot at 0,0

fig,ax = plt.subplots(1,1,figsize=(1,1),dpi=150)
conditions = {'audDiff': 0, 'visDiff': 0, 'isOpto': 0}
                        
plots.plot_diagnostics(model=None,sample=sample,
    conditions=conditions,
    data_plot_func='plot',
    dkwargs={'color':'k','alpha':1,'linewidth':.7,'markersize':2},
    data_dt=.035,ax=ax,tlim=[0,.85])


ax.axhline(0, color='black', lw=.5, linestyle=':')
ax.set_xlabel('RTs (s)')
ax.set_ylabel('Pdf.')

#fig.savefig(SAVE_PATH + r'\ctrl_model_diagnostics_0_0.svg',dpi=300,bbox_inches='tight')


#%% 
from src.read_model import load_saved_model

m_type  = 'av_lazy'
model_path = np.array(dataset.model_paths)[np.array(dataset.model_types)==m_type][0]

#%%
m = load_saved_model(model_path,model_class=m_type)

fig = plots.plot_av_diagnostics2(sample, model=m, plot_ctrl=True, plot_opto=False,xlims=[.05,.5],
                           ctrl_kws={'color':'k','alpha':.1,'linewidth':.5},
                           mkwargs={'color':'k','alpha':.7,'lw':.6,'markersize':1})


fig.savefig(SAVE_PATH + r'\ctrl_distributions_model_{}.svg'.format(m_type),dpi=300,bbox_inches='tight')


#%% diagnostics across conditions

import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,1,figsize=(.9,2.3),sharex=True,sharey=False,gridspec_kw={'height_ratios':[2,1,2]})
fig.subplots_adjust(hspace=0.1)

plots.plot_psychometric(m,sample=sample,axctrl=ax[0],axopto=None,plot_opto=False,datkws={'s':10,'edgecolor':'k','linewidth':0.5})
plots.plot_undecided(sample,model = m,axctrl=ax[1],axopto=None,plot_opto=False,datkws={'marker':'.','linestyle':'None','markersize':8,
                                                                                       'markeredgecolor':'k'})
plots.plot_chronometric(m,sample=sample,axctrl=ax[2],axopto=None,plot_opto=False,datkws={'s':10,'edgecolor':'k','linewidth':0.5})


ax[0].set_ylim([0,1])
ax[1].set_ylim([0,0.17])
ax[2].set_ylim([0.25,0.5])

ax[0].set_yticks([0,1])
ax[1].set_yticks([0.05,0.15])
ax[2].set_yticks([0.3,0.4,0.5])


for i, a in enumerate(ax):
    a.set_xticks([-1,0,1])
    a.set_title('')
    
    if i == 2:
        a.set_xlabel('Contrast (a.u.)')

ax[0].set_ylabel('p(Right)')
ax[1].set_ylabel('p(NoGo)')
ax[2].set_ylabel('median RT (s)')
fig.savefig(SAVE_PATH + r'\ctrl_diagnostics_model_{}.svg'.format(m_type),dpi=300,bbox_inches='tight')


# %%
