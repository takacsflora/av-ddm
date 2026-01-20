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

which = 'SC_bi'
path = Path(rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\{which}\with_undecided\opto')
dataset = get_dataset(path)


#%%
dataset = dataset[dataset.stem=='summary_sample'].iloc[0]

sample = read_pickle(dataset.train_path)

#%%
fig = plots.plot_av_diagnostics2(sample, model=None, plot_ctrl=True, plot_opto=True,xlims=[.05,.5],
                           ctrl_kws={'color':'k','alpha':1,'linewidth':.75,'markersize':1.5},
                           opto_kws={'color':'#4EA72E','alpha':1,'linewidth':.75,'markersize':1.5},mkwargs={})

fig.savefig(SAVE_PATH + rf'\opto_diagnostics_{which}_data_only.svg',dpi=300,bbox_inches='tight')

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

if which == 'SC_bi':
    m_type  = 'av_lazy_opto_bilateral'
    p_set = 'g_d_Bound'  # model parameter set to load
elif which == 'SC_uni':
    m_type  = 'av_opto_unilateral'
    p_set = 'g_d_bBL'  # model parameter set to load



model_path = np.array(dataset.model_paths)[(np.array(dataset.model_types)==m_type) & 
                                           (np.array(dataset.params_tested)==p_set)][0]


m = load_saved_model(model_path,model_class=m_type)

#%%
fig = plots.plot_av_diagnostics2(sample, model=m, plot_ctrl=False, plot_opto=True,xlims=[.05,.5],
                           ctrl_kws={'color':'k','alpha':.1,'linewidth':.5},
                           opto_kws={'color':'#4EA72E','alpha':.7,'lw':.6},
                           mkwargs={'color':'k','alpha':.7,'lw':.6,'markersize':1})


fig.savefig(SAVE_PATH + rf'\opto_RTdists_{which}_model_{m_type}_{p_set}.svg',dpi=300,bbox_inches='tight')

# %%


import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,2,figsize=(1.5,2),sharex=True,sharey=False,gridspec_kw={'height_ratios':[2,1,2]})
ax[0,0].figure.subplots_adjust(hspace=0.1)  # Adjust space between row 0 and row 1
ax[1,0].figure.subplots_adjust(hspace=0.2)  # Adjust space between row 1 and row 2

fig.subplots_adjust(wspace=0.2)  # Adjust space between columns

plots.plot_psychometric(m,sample=sample,axctrl=ax[0,0],axopto=ax[0,1],plot_opto=True,datkws={'s':10,'edgecolor':None,'linewidth':0.5})
plots.plot_undecided(sample,model = m,axctrl=ax[1,0],axopto=ax[1,1],plot_opto=True,datkws={'marker':'.','linestyle':'None','markersize':5,
                                                                                       'markeredgecolor':'k','linewidth':0.5,'markeredgewidth':0.5})
plots.plot_chronometric(m,sample=sample,axctrl=ax[2,0],axopto=ax[2,1],plot_opto=True,datkws={'s':10,'edgecolor':'k','linewidth':0.5})



for rows in ax:
    for i, a in enumerate(rows):
        a.set_xticks([-1,0,1])
        a.set_title('')
        
        if i == 2:
            a.set_xlabel('Contrast (a.u.)')
        else:
            a.set_xlabel('')


for colID in range(2): 
    ax[0,colID].set_ylim([0,1])
    ax[1,colID].set_ylim([0,0.4])
    ax[2,colID].set_ylim([0.20,0.45])
    ax[0,colID].set_yticks([0,1])
    ax[1,colID].set_yticks([0.1,0.3])
    ax[2,colID].set_yticks([0.2,0.4])

    if colID == 0:
        ax[0,colID].set_ylabel('p(Right)')
        ax[1,colID].set_ylabel('p(NoGo)')
        ax[2,colID].set_ylabel('median RT (s)')
    else:
        ax[0,colID].set_ylabel('')
        ax[1,colID].set_ylabel('')
        ax[2,colID].set_ylabel('')
    
    if colID ==1: 
        ax[0,colID].set_yticklabels([])
        ax[1,colID].set_yticklabels([])
        ax[2,colID].set_yticklabels([])

fig.savefig(SAVE_PATH + rf'\opto_diagnostics_{which}_model_{m_type}_{p_set}.svg',dpi=300,bbox_inches='tight')
# %%
