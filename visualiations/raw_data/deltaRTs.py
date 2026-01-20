# %% 
# to plot average delta RTs for each model

from utils.av_dat_manager import read_csvs,preproc_ev,filter_ev
import pandas as pd


def get_average_delta_RT(ev): 
    grouped = ev.groupby(['visDiff', 'audDiff', 'choice', 'is_laserTrial'])['RT'].median().reset_index()

    grouped['cond'] = grouped['visDiff'].astype(str) + '_' + grouped['audDiff'].astype(str) + '_' + grouped['choice'].astype(str)

    grouped = grouped.pivot(index='cond', columns=['is_laserTrial'], values=['RT']).reset_index()

    grouped.columns = ['cond', 'RT_noLaser', 'RT_laser']

    grouped['deltaRT'] = grouped['RT_laser'] - grouped['RT_noLaser']
    grouped['choice'] = grouped['cond'].apply(lambda x: x.split('_')[-1]).astype('int')


    average_deltaRT = grouped.groupby('choice')['deltaRT'].mean().reset_index()

    return average_deltaRT




sets = ['uni_SC_nogo','bi_SC_nogo','uni_MOs_nogo','bi_MOs_nogo']
rt_rel_to = 'stim'

dRTs = []
for set_name in sets:
    ev = read_csvs(set_name = set_name)
    ev = preproc_ev(ev)
    ev = filter_ev(ev,rt_rel_to = rt_rel_to)

    unique_subjects = ev['file'].unique()
    for subject in unique_subjects:
        deltaRT = get_average_delta_RT(ev[ev['file'] == subject])
        deltaRT['set'] = set_name
        deltaRT['subject'] = subject

        dRTs.append(deltaRT)

dRTs = pd.concat(dRTs,ignore_index=True)


#%%
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 8

fig, ax = plt.subplots(1, len(sets), figsize=(3, 1.5),sharex=True, sharey=True)
fig.subplots_adjust(wspace=.3)
set_name_labels = ['SC$_{uni}$', 'SC$_{bi}$', 'MOs$_{uni}$', 'MOs$_{bi}$']

for i, set_name in enumerate(sets):
    for subject in dRTs[dRTs['set'] == set_name]['subject'].unique():
        sns.lineplot(
            data=dRTs[(dRTs['set'] == set_name) & (dRTs['subject'] == subject)],
            x='choice',
            y='deltaRT',
            color='grey',
            alpha=0.5,
            ax=ax[i],
            legend=False
        )

    sns.lineplot(
        data=dRTs[dRTs['set'] == set_name].groupby(['choice']).mean().reset_index(),
        x='choice',
        y='deltaRT',
        color='black',
        linewidth=2,
        ax=ax[i],
        legend=False
    )
    ax[i].set_title(set_name_labels[i])

    ax[i].set_ylim(-0.2, 0.2)
    ax[i].set_yticks([-0.2, 0, 0.2])

    ax[i].set_yticklabels(['-.2', '0', '.2'])
    ax[i].tick_params(axis='x', rotation=0)

    sns.despine(ax=ax[i], top=True, right=True)

    ax[i].set_xlabel('')
    ax[i].set_ylabel('ΔRT (s), opto — control')
    ax[i].set_xticks([0, 1])
    ax[i].set_xticklabels(['L', 'R'])

    ax[i].axhline(0, color='black', linestyle='--', linewidth=1)



# %%

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15

fig,ax = plt.subplots(1,1,figsize=(3,3))

sns.barplot(data=dRTs,x='set',y='deltaRT',
            hue='choice',
            palette=['magenta','green'],dodge=True,ax=ax,legend=False)

ax.set_ylabel('ΔRT (ms)')
ax.set_yticks([-0.04,-0.02,0,0.02])
ax.set_yticklabels([-40,-20,0,20])

plt.rcParams['font.size'] = 15


ax.set_xticklabels(['SC$_{uni}$', 'SC$_{bi}$', 'MOs$_{uni}$', 'MOs$_{bi}$'])
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
sns.despine(ax=ax, top=True, right=True)
# %%
