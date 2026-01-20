#%%
from pathlib import Path
from utils.add_src_to_sys import *
from src.evaluate import load_evaluation,get_gain_loss_models

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# fix this 

path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\SC_bi\with_undecided\opto')
df = load_evaluation(path,recompute = True)
# %%

# test whether 
