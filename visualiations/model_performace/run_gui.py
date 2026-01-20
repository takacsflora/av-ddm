#%%
from utils.add_src_to_sys import *


import pyddm 
from utils.av_dat_manager import read_csvs,preproc_ev,filter_ev


# get the sample
df = read_csvs(set_name = 'uni_SC_nogo')
df = preproc_ev(df)
df = filter_ev(df,rt_rel_to = 'stim',ctrl_only=False)

sample = pyddm.Sample.from_pandas_dataframe(df, 
                                            rt_column_name="RT", 
                                            choice_column_name="choice", 
                                            choice_names =  ("Right", "Left"))

#
#%%
# assemble the model
import src.AVmodel as AV

params = AV.all_params()
m = AV.assemble_model(params)



import pyddm.plot
#%%
pyddm.plot.model_gui(model = m ,sample = sample)