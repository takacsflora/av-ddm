#%%

# I don't know what is happening with the batch fit so I am need to look at the individual fits
from pathlib import Path
from utils.add_src_to_sys import *

from src.my_io import read_pickle
from src.evaluate import get_dataset
from src.read_model import get_model



from src.AVmodel import get_param_sets

ps = get_param_sets()
#%%
path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim')
dataset = get_dataset(path)
dataset = dataset[dataset.stem=='AV036_right_1mW'].iloc[0]

train_sample = read_pickle(dataset.train_path)


# %%

model = get_model(dataset.model_paths[1])


# %%
import plots

fig = plots.av_diagnostics(train_sample,model = model)


# %%
