#%%
# 
import pyddm
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from utils.add_src_to_sys import *
import src.AVmodel as AV
import pyddm



SAVE_PATH = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\DDM'


plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

#%%
params = {'aR': 1.65,
 'vR': 1.68,
 'aL': 1.24,
 'vL': 1.42,
 'a_pulse_end': .35,
 'gamma': .75,
 'b': -.26,
 'x0': -0.03,
 'nondec': .14,
 'lazy_rate':1.6, 
 'umixture': 0,
 'B': .41,
#  'd_aR': 0,
#  'd_vR': 0,
#  'd_aL': 0,
#  'd_vL': 0,
 'd_b':0,
 'd_x0': 0,
 'd_nondec': 0,
 'd_lazy_rate': 0,
 'd_BL': 0.1}


#%%

params = {
    'aR': 1.65,
    'vR': 1.68,
    'aL': 1.24,
    'vL': 1.42,
    'a_pulse_end':.35,
    "gamma":.75,
    'bias': -0.26,
    'x0': -0.03,
    'nondec': 0.14,
    'lazy_rate':1.6, 
    'umixture': 0,
    'B': .41,
    'd_bias': 0,
    'd_x0': 0,
    'd_nondec': 0,
    "d_Bound": .07,
    'd_lazy_rate': 0
}

#%%
modelfunctions,_,hyperparams = AV.get_model(which='av_lazy_opto_bilateral')
#%%
m = pyddm.gddm(parameters=params,**modelfunctions,**hyperparams) 


#%%
# %


def get_median(pdf_norm):
    median_value = (np.cumsum(pdf_norm).max() / 2)
    median_loc = np.argmin(np.abs((np.cumsum(pdf_norm) - median_value)))
    return median_value,median_loc

visDiffs = np.linspace(-1,1,5)
visDiffs = np.array([0])
n_plots = visDiffs.size
fig,axs = plt.subplots(1,visDiffs.size,figsize=(n_plots*.5,.5),sharey=True,dpi=150)

fig.patch.set_alpha(0)
right_choices,undecideds = [],[]

plot_cumulative = False  

nogox = 0.45
colors = ['black','#4EA72E']
for vidx,visDiff in enumerate(visDiffs):
    v_R = np.abs(visDiff)*(visDiff>0)
    v_L = np.abs(visDiff)*(visDiff<0)
    
    if n_plots ==1:
        axu = axs
    else:
        axu = axs[vidx]
    for i in range(2):
        conditions = {'a_L':0, 'a_R':0, 'v_R':v_R, 'v_L':v_L, 'isOpto':i}

        sol = m.solve(conditions=conditions)

        pdf_r = sol.pdf('Right')
        #axu.plot(m.t_domain(),(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])

        if plot_cumulative:
            axu.plot(m.t_domain(),np.cumsum(pdf_r),color=colors[i])
            median_value, median_loc = get_median(pdf_r)
            axu.plot(m.t_domain()[median_loc], median_value, '.', color=colors[i])
            axu.vlines(m.t_domain()[median_loc],median_value,0, color=colors[i])
        else:
            axu.plot(m.t_domain(),pdf_r,color=colors[i],linewidth=.75)
        pdf_l = sol.pdf('Left')

        if plot_cumulative: 
            axu.plot(m.t_domain(),-np.cumsum(pdf_l),color=colors[i])
            median_value, median_loc = get_median(pdf_l)
            axu.plot(m.t_domain()[median_loc], -median_value, '.', color=colors[i])
            axu.vlines(m.t_domain()[median_loc],-median_value,0, color=colors[i])
        else:
            axu.plot(m.t_domain(),-pdf_l,color=colors[i],linewidth=.75)

        p_undecided = sol.prob_undecided()
        axu.plot([nogox,nogox],[0, p_undecided/0.025],'-',alpha=1,markersize=2,color=colors[i],linewidth=.75)
        axu.plot(nogox, p_undecided/0.025,'o',alpha=1,markersize=2,color=colors[i])

        axu.plot([nogox,nogox],[0, -p_undecided/0.025],'-',alpha=1,markersize=2,color=colors[i],linewidth=.75)
        axu.plot(nogox, -p_undecided/0.025,'o',alpha=1,markersize=2,color=colors[i])
        
        undecideds.append({'visDiff': visDiff, 'isOpto': i, 'p_undecided': p_undecided})

        p_right = sol.prob(choice='Right')
        right_choices.append({'visDiff': visDiff, 'isOpto': i, 'p_right': p_right})

        axu.axhline(0, color='k', linestyle=':', alpha=0.5,linewidth=0.5)

        axu.set_xlim([0.1,0.5])

    #axu.set_title(f'visDiff={visDiff:.2f}')

undecided_df = pd.DataFrame(undecideds)
right_choices_df = pd.DataFrame(right_choices)

if n_plots ==1:
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.set_yticks([])
    axs.set_xticks([])
    axs.set_ylim([-10,10])
else:
    for ax in axs:
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

fig.savefig(SAVE_PATH + r'\opto_model_simulated_distributions_d_lazy_bilateral.svg',dpi=300,bbox_inches='tight')



# %%
import seaborn as sns
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
sns.lineplot(data=undecided_df, x='visDiff', y='p_undecided', hue='isOpto',palette=colors,legend=False,ax=ax)# %

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylim(-.02, .57)
# %%
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)

sns.lineplot(data=right_choices_df, x='visDiff', y='p_right', 
                hue='isOpto',palette=colors,legend=False,ax=ax)# %%


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# %%
