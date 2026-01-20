#%%

# I don't know what is happening with the batch fit so I am need to look at the individual fits
from pathlib import Path
from utils.add_src_to_sys import *

from src.my_io import read_pickle
from src.evaluate import get_dataset,get_logliks,get_gain_loss_models
from src.read_model import load_saved_model

from src.AVmodel import get_model

ps = get_model(which='av_lazy')
#%%
import matplotlib.pyplot as plt
import numpy as np
path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\sumdat_with_undecided')
dataset = get_dataset(path)

#%
dataset = dataset[dataset.stem=='uni_SC_nogo_Sample_all'].iloc[0]
# result = get_logliks(dataset)
# result = get_gain_loss_models(result)

# #%%
# import seaborn as sns
# sns.lineplot(data=result,x='param',y='LogLik_test',hue='type',markers=True)

# plt.xticks(rotation=90)

#%%
train_sample = read_pickle(dataset.train_path)
#test_sample = read_pickle(dataset.test_path)


model_type = 'g_d_bBL'
model_path = np.array(dataset.model_paths)[np.array(dataset.model_names)==model_type][0]



m = load_saved_model(model_path,model_name='av_lazy')


model_params = {mym:v for mym,v in zip(list(m.get_model_parameter_names()),list(m.get_model_parameters()))}



# %%
import plots




fig = plots.av_diagnostics(train_sample,model = m)

#%%
fig,axs = plt.subplots(1,2,figsize=(4,2),sharey=True, sharex=True)

for axis in axs:
    for spine in ['top', 'right']:
        axis.spines[spine].set_visible(False)

plots.plot_undecided(train_sample,model = m,axctrl=axs[0],axopto=axs[1])
axs[1].set_ylabel('')

#%%
fig,axs = plt.subplots(1,2,figsize=(4,2),sharey=True, sharex=True)

for axis in axs:
    for spine in ['top', 'right']:
        axis.spines[spine].set_visible(False)

plots.plot_psychometric(m,sample=train_sample,axctrl=axs[0],axopto=axs[1])
axs[1].set_ylabel('')



fig,axs = plt.subplots(1,2,figsize=(4,2),sharey=True, sharex=True)

for axis in axs:
    for spine in ['top', 'right']:
        axis.spines[spine].set_visible(False)

plots.plot_chronometric(m,sample=train_sample,axctrl=axs[0],axopto=axs[1])

axs[1].set_ylabel('')






# %%
plots.plot_av_diagnostics2(train_sample, model=None, plot_ctrl=True, plot_opto=False,xlims=[.03,.8])
# %%
