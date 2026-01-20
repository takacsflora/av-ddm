#%%


#%% 
import numpy as np 
import pandas as pd
from pathlib import Path


from utils.add_src_to_sys import *
from src.evaluate import load_evaluation


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 8,'font.family':'Calibri','axes.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

# fix this 
SAVE_PATH = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\DDM'
path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\all_mice\with_undecided\ctrl')
df = load_evaluation(path,recompute = False)



# %%

# plot logLik_test per model_types


df_melted = df.melt(id_vars=['model_type','subject'],
                    value_vars=['LogLik_test','LogLik_train'],
                    var_name='logLik_type',
                    value_name='logLik')


df_melted_test = df_melted[df_melted['logLik_type']=='LogLik_test'].copy()
df_melted_test['model_type'] = df_melted_test['model_type'].replace({'av_original': 'uniform', 'av_lazy': 'lazy'})

# plot each subject's line per model type in grey lines

fig,ax = plt.subplots(1,1,figsize=(.4,.8),dpi=150)

subjects = df_melted_test['subject'].unique()
df_melted_test['model_type'] = pd.Categorical(df_melted_test['model_type'], categories=['uniform', 'lazy'], ordered=True)
for subject in subjects:
    sns.lineplot(data=df_melted_test[df_melted_test['subject']==subject],
                 x='model_type', y='logLik', color='grey', alpha=0.3, legend=False,ax=ax)


                # plot mean logLik per model type in black line
mean_logLik = df_melted_test.groupby('model_type')['logLik'].mean()
sns.lineplot(data=mean_logLik.reset_index(), x='model_type', y='logLik', color='black', linewidth=3, ax=ax)

ax.set_ylabel('- Log$_{10}$Likelihood \n (test)')
ax.set_xlabel('')
ax.set_xticklabels(['Uniform', 'Lazy'],rotation=45)
#sns.barplot(data=df_melted_test, x='model_type', y='logLik', legend=False)
fig.savefig(SAVE_PATH + r'\ctrl_model_logLik_test.svg',dpi=300,bbox_inches='tight')

# %%
