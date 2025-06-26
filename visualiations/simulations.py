#%%
# 
import pyddm
import numpy as np 
import matplotlib.pyplot as plt

from utils.add_src_to_sys import *

import matplotlib.pyplot as plt


import src.AVmodel as AV
import pandas as pd


#%%

params = {'aR': 0,
 'vR': 1,
 'aL': 0,
 'vL': 1,
 'gamma': 1,
 'b': 0,
 'x0': 0,
 'nondec':0.1,
 'noise': 2,
 'umixture':0.01,
 'BL': 1,
 'BR': 1,
 'd_aR': 0,
 'd_vR': 0,
 'd_aL': 0,
 'd_vL': 0,
 'd_b': 3,
 'd_x0': 0,
 'd_nondec': 0,
 'd_BL': 0}

#

m= AV.assemble_model(params,drift_type='av')
# %

visDiffs = np.linspace(-1,1,5)
fig,axs = plt.subplots(1,visDiffs.size,figsize=(8,2),sharey=True,dpi=150)

undecideds = []

colors = ['black','orange']
for vidx,visDiff in enumerate(visDiffs):
    v_R = np.abs(visDiff)*(visDiff>0)
    v_L = np.abs(visDiff)*(visDiff<0)
    axu = axs[vidx]
    for i in range(2):
        conditions = {'a_L':0, 'a_R':0, 'v_R':v_R, 'v_L':v_L, 'isOpto':i}

        sol = m.solve(conditions=conditions)

        pdf_r = sol.pdf('Right')
        #axu.plot(m.t_domain(),(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])
        axu.plot(m.t_domain(),pdf_r,color=colors[i])

        pdf_l = sol.pdf('Left')

        axu.plot(m.t_domain(),-pdf_l,color=colors[i])

        p_undecided = sol.prob_undecided()
        axu.plot(m.T_dur, p_undecided,'*',alpha=1,markersize=10,color=colors[i])
        undecideds.append({'visDiff': visDiff, 'isOpto': i, 'p_undecided': p_undecided})

    axu.set_title(f'visDiff={visDiff:.2f}')

undecided_df = pd.DataFrame(undecideds)

# %%
import seaborn as sns
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
sns.scatterplot(data=undecided_df, x='visDiff', y='p_undecided', hue='isOpto',palette=colors,legend=False,ax=ax)
# %%
