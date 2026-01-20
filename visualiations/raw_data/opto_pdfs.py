#%%


from utils.add_src_to_sys import *

import src.AVmodel as AV
import src.undecided as Undectools

import pyddm 
import numpy as np
from utils.av_dat_manager import get_summary_dataset
from visualiations.model_performace import plots

# get the sample
dataset_name = 'uni_MOs_nogo'
sample = get_summary_dataset(dataset_name, recompute=False, subsample=True, 
                         rt_rel_to='stim',keep_undecided = True, ctrl_only = False, max_rt = 1.5)
#%
fig = plots.ctrl_vs_opto(sample)

plots.plot_undecided(sample)


plots.plot_undecided_ctrl_vs_opto(sample)


plots.plot_av_diagnostics2(sample, plot_ctrl=True, plot_opto=True,xlims=[.05,.5])
# %%
