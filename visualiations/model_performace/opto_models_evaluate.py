#%% 
import numpy as np 
import pandas as pd
from pathlib import Path


from utils.add_src_to_sys import *
from src.evaluate import load_evaluation,get_gain_loss_models

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel



SAVE_PATH = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\DDM'


plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})



# %%
region = 'SC'
hemisphere = 'bi'

# fix this 

path = Path(rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\{region}_{hemisphere}\with_undecided\opto')
df = load_evaluation(path,recompute = False)

#%%
df['hemisphere'] = df['hemisphere'].replace({'right': 'uni', 'left': 'uni'})

df_logLik = get_gain_loss_models(df)


#df_loglik  = df_logLik[df_logLik.subject.isin(['''AV036','AV038','AV041','AV044','AV047','AV055'])].copy()



#%%
df_logLik_full = df_logLik[
                        (df_logLik['inactivation_location'] == region) & 
                        (df_logLik['hemisphere'] == hemisphere) & 
                        (df_logLik['param']=='ctrl')
].copy()

fig, ax = plt.subplots(1, 1, figsize=(.4, .4),dpi=150)
                       
# Loop over stems and plot in grey

statistic = 'LogLik_test'

for stem in df_logLik_full['stem'].unique():
    sns.lineplot(
        data=df_logLik_full[df_logLik_full['stem'] == stem],
        x='type', y='LogLik_test', color='grey', alpha=0.5, legend=False,ax=ax
    )

# Plot the average in black
df_avg = df_logLik_full.groupby('type', as_index=False)[statistic].mean()
sns.lineplot(data=df_avg, x='type', y=statistic, color='black', linewidth=2, legend=False,ax=ax)


# Perform a paired t-test between 'ctrl' and 'full'
ctrl_values = df_logLik_full[df_logLik_full['type'] == 'gain'][statistic]
full_values = df_logLik_full[df_logLik_full['type'] == 'loss'][statistic]

t_stat, p_value = ttest_rel(ctrl_values, full_values)

print(f"Paired t-test results: t-statistic = {t_stat:.3f}, p-value = {p_value:.3e}")


# Rename xticks from 'gain' and 'loss' to 'ctrl' and 'full opto model'
ax.set_xticks([0, 1])
ax.set_xticklabels(['ctrl', 'opto (full)'])

# Turn off all spines except for the y-axis
sns.despine(ax=ax, left=False, right=True, top=True, bottom=True)
ax.set_ylabel('-Log₁₀Likelihood \n (test)')
ax.set_xlabel('')
ax.set_yticks([-.8, -.4, 0])
ax.set_xlim([-.2,1.1])

fig.savefig(SAVE_PATH + rf'\opto_model_LogLikelihood_{region}_{hemisphere}_full_vs_ctrl.svg',dpi=300,bbox_inches='tight')

#%%
#sel_params = df_logLik.param.unique()

#sel_params = ['d_nondec','d_aR','d_aL','d_vL','d_vR','d_BL','d_x0','d_b']
if hemisphere == 'uni':
    sel_params = ['ctrl','d_nondec', 'd_BL','d_x0', 'd_b', 'd_x0BL', 'd_bx0','d_bBL']
        #'d_bx0BL']
    sel_params_names = ['ctrl', 'Δ tND', 'Δ Bound', 'Δ x0', 'Δ bias', 'Δ x0 + Bound', 'Δ bias + x0', 'Δ bias + Bound']
elif hemisphere == 'bi':
    sel_params = ['ctrl','d_nondec','d_lazy_rate', 'd_x0', 'd_bias','d_Bound']
    sel_params_names = ['ctrl', 'Δ tND', 'Δ Lazy rate', 'Δ x0', 'Δ bias', 'Δ Bound']
               #'d_Boundlazy', 'd_x0Bound', 'd_biasBound']
            # Update parameter names to include proper delta Greek string
            # Replace underscores with spaces and ensure proper formatting for parameter names
# sel_params_names = [r'control', r'$\Delta t_{ND}$', r'$\Delta x_0$', r'$\Delta bias$', 
#                     r'$\Delta Bound$', r'$\Delta lazy\ rate$']


                   # 'Δ Bound + lazy_rate', 'Δ x0 + Bound', 'Δ bias + Bound']
#sel_params_names = ['ctrl', r'$t_{ND}$', r'$x_0$', 'bias', 'Bound', 'Lazy rate']

df_logLik_region = df_logLik[
                        (df_logLik['inactivation_location'] == region) & 
                        (df_logLik['hemisphere'] == hemisphere) & 
                        np.isin(df_logLik['param'],sel_params) 
].copy()


fig,ax = plt.subplots(1,1,figsize=(.7,.7), dpi=150)

metric = 'LogLik_test_norm'
# sns.lineplot(data=df_logLik_region, x='param', y=metric, hue='type',
#               style='stem', markers=True, dashes=False,legend=False,ax=ax,alpha=.2)
df_logLik_region['param'] = pd.Categorical(df_logLik_region['param'], 
                                           categories=sel_params, ordered=True)
sns.lineplot(data=df_logLik_region, x='param', y=metric, hue='type',
                 markers=True, dashes=True,legend=False,ax=ax,alpha=1,linewidth=1.5)



ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.axhline(1,linestyle=':',color='grey')
ax.axhline(0,linestyle=':',color='grey')

ax.set_xlabel('')

sns.despine(ax=ax, left=False, right=True, top=True, bottom=True)

ax.set_ylabel('Δ Log₁₀Likelihood \n (norm.)')
ax.set_yticks([0,1])
ax.set_ylim([-0.2,1.2])
ax.set_xticklabels(sel_params_names, rotation=90)#


fig.savefig(SAVE_PATH + rf'\opto_model_comparison_{region}_{hemisphere}.svg',dpi=300,bbox_inches='tight')


#ax.tick_params(axis='x', bottom=True, top=True, labelbottom=True, labeltop=True)

# %%
# now we plot the parameters of the full model


df_full = df[
    (df['inactivation_location'] == region) & 
    (df['hemisphere'] == hemisphere) & 
    (df['paramset'] == 'full')  # Change to your model name
    ].copy()


#params = ['d_b','d_BL','d_nondec','d_x0']
params = ['d_nondec','d_Bound','d_x0','d_bias']

n_params = len(params)
fig,ax = plt.subplots(1,n_params,figsize=(n_params*2,3),sharex=True,sharey=True)
fig.subplots_adjust(wspace=1)

for i,param in enumerate(params): 
    ax[i].plot(df_full[param],'o')
    ax[i].axhline(0,linestyle='--',color='grey')
    ax[i].set_title(param)


# %%

df['inactivation_type'] = df['inactivation_location'] + '_' + df['hemisphere']


df_full = df[df.paramset == 'g_d_Bound'].copy()

df_melted = df_full.melt(
    id_vars=['inactivation_type'], 
    value_vars=params, 
    var_name='param', 
    value_name='param_value'
)

# Plot param name against value with inactivation_type as hue
fig, ax = plt.subplots(figsize=(12, 4))
sns.swarmplot(data=df_melted, x='param', y='param_value',
               hue='inactivation_type', 
              ax=ax,dodge=True)
ax.axhline(0, linestyle='--', color='grey')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Parameter Values by Inactivation Type')
plt.legend(title='Inactivation Type')
plt.tight_layout()

# %%

sns.pairplot(df_full[['d_Bound','d_bias']])

# %%
