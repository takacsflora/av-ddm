#%% 
import numpy as np 
import pandas as pd
from pathlib import Path
from utils.add_src_to_sys import *
from src.evaluate import load_evaluation,get_gain_loss_models

import matplotlib.pyplot as plt
import seaborn as sns

path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim')
df = load_evaluation(path,recompute = False)

df['hemisphere'] = df['hemisphere'].replace({'right': 'uni', 'left': 'uni'})

df_logLik = get_gain_loss_models(df)



# %%
region = 'SC'
hemisphere = 'uni'
#sel_params = df_logLik.param.unique()

sel_params = ['d_nondec','d_aR','d_aL','d_vL','d_vR','d_x0','d_b']

df_logLik_region = df_logLik[
                        (df_logLik['inactivation_location'] == region) & 
                        (df_logLik['hemisphere'] == hemisphere) & 
                        np.isin(df_logLik['param'],sel_params) 
].copy()


fig,ax = plt.subplots(1,1,figsize=(3,2))

metric = 'LogLik_test_norm'
# sns.lineplot(data=df_logLik_region, x='param', y=metric, hue='type',
#               style='stem', markers=True, dashes=False,legend=False,ax=ax,alpha=.2)
df_logLik_region['param'] = pd.Categorical(df_logLik_region['param'], 
                                           categories=sel_params, ordered=True)
sns.lineplot(data=df_logLik_region, x='param', y=metric, hue='type',
                 markers=True, dashes=False,legend=False,ax=ax,alpha=1)


ax.set_ylim([-.5,1.7])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.axhline(1,linestyle='--',color='grey')
ax.axhline(0,linestyle='--',color='grey')

ax.set_xlabel('')

sns.despine(ax=ax, left=False, right=True, top=True, bottom=True)

ax.set_ylabel('Δ Performance, norm.')
ax.set_yticks([0,1])
#ax.tick_params(axis='x', bottom=True, top=True, labelbottom=True, labeltop=True)

# %%
# now we plot the parameters of the full model


df_full = df[
    (df['inactivation_location'] == region) & 
    (df['hemisphere'] == hemisphere) & 
    (df['model_name'] == 'full')
    ].copy()


params = ['d_nondectimeOpto']

n_params = len(params)
fig,ax = plt.subplots(1,n_params,figsize=(n_params*2,3),sharex=True,sharey=True)
fig.subplots_adjust(wspace=1)

for i,param in enumerate(params): 
    ax[i].plot(df_full[param],'o')
    ax[i].axhline(0,linestyle='--',color='grey')
    ax[i].set_title(param)


# %%




df['inactivation_type'] = df['inactivation_location'] + '_' + df['hemisphere']


df_full = df[df.model_name == 'full'].copy()

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
