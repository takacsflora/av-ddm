

# %%
import pyddm 
import pyddm.plot
from src.fitting import get_parameters
from src.model_components import get_freeP_sets
from utils.av_dat_manager import read_csvs


df = read_csvs(set_name = 'bi_MOs_nogo')


df = df[((df.choice==0) | (df.choice==1)) & (~df.RT.isna())] # remove NaN and RTs > 2s
#%%
sample = pyddm.Sample.from_pandas_dataframe(df, 
                                            rt_column_name="RT", 
                                            choice_column_name="choice", 
                                            choice_names =  ("Right", "Left"))

freePs = get_freeP_sets('all')
fit_params = get_parameters(freePs=freePs)             
m = pyddm.Model(**fit_params)




#sample = read_pickle(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\samples\train\AV036_10mW_Left_Sample_train.pickle')
pyddm.plot.model_gui(model = m ,sample = sample)

