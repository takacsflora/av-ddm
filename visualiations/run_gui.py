

# %%
from utils.add_src_to_sys import *

import pyddm 
import pyddm.plot
from src.fitting import get_parameters
from src.model_components import get_freeP_ctrl,get_freeP_opto
from utils.av_dat_manager import read_csvs,preproc_ev,filter_ev


df = read_csvs(set_name = 'bi_MOs_nogo')
df = preproc_ev(df)
df = filter_ev(df,rt_rel_to = 'stim',ctrl_only=False)


#%%
sample = pyddm.Sample.from_pandas_dataframe(df, 
                                            rt_column_name="RT", 
                                            choice_column_name="choice", 
                                            choice_names =  ("Right", "Left"))

freePs = get_freeP_opto('full')
fit_params = get_parameters(fit_type='opto',freePs=freePs)             
m = pyddm.Model(**fit_params)




#sample = read_pickle(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\samples\train\AV036_10mW_Left_Sample_train.pickle')
pyddm.plot.model_gui(model = m ,sample = sample)

