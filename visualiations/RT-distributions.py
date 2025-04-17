

# %%
import pyddm 
from utils.av_dat_manager import read_csvs
import plots

df = read_csvs(set_name = 'uni_SC_nogo')

min_trials_per_file = df['file'].value_counts().min()
df = df.groupby('file', group_keys=False).apply(lambda x: x.sample(n=min_trials_per_file, random_state=42))

# keep also choices that have been already moving at stimulus onset? 

#%%
df = df[((df.choice==0) | (df.choice==1)) & (~df.RT.isna())] # remove NaN and RTs > 2s

sample = pyddm.Sample.from_pandas_dataframe(df, 
                                            rt_column_name="RT", 
                                            choice_column_name="choice", 
                                            choice_names =  ("Right", "Left"))


fig = plots.av_diagnostics(sample,model=None)

#%%
# fit a model on our sample if we want? 

# for just the drift gain model this takes 10 mins so, can take a while!!!
# from src.fitting import get_parameters
# from src.model_components import get_freeP_sets

# freePs = get_freeP_sets('g_d_b')
# fit_params = get_parameters(freePs=freePs)             
# m = pyddm.Model(**fit_params)
# pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossLikelihood, verbose=False) 

# %%


# this is way too many plots, lets try to focus on a few  values

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 15

aud_azimuths = [-60,0,0,0,60]
vis_contrasts =  [-1,-1,0,1,1]

fig,ax = plt.subplots(1,len(aud_azimuths),figsize=(9,4.5),sharey=True, sharex=True)

colors = ['purple','orange']
scaling_factor = 3
for isLaser in range(2):
    for i,(a,v) in enumerate(zip(aud_azimuths,vis_contrasts)):
            curr_cond ={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
            plots.plot_diagnostics(model=None,sample = sample, 
                            conditions=curr_cond,data_dt =.025,method=None,myloc=0,ax = ax[i],
                            dkwargs={'color':colors[isLaser],'alpha':.7,'linewidth':1},
                            mkwargs = {'color':colors[isLaser],'alpha':1,'linewidth':3},
                            time_on_x=True,data_plot_func='fill_between')
           
ax[0].set_xlim([.1,.7])
# %%
