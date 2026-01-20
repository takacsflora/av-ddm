
#%%
import pyddm
from AVmodel import get_model,get_delta_param_sets
from my_io import read_pickle
from pathlib import Path



def construct_model_path(sample_name = 'uni_SC_nogo', model_name = 'g_d_b', path = None):

    if path == '':
        path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\summary_data')
    else:
        path = Path(path)

    path = path / 'model_fits'
    
    file_path = path / f'{sample_name}_Model_{model_name}.pickle'

    return file_path



def load_saved_model(file_path = None, model_class = 'av_lazy',**kwargs): 
    """
    function to load a model from a given path. 

    Args:
        path (str, optional): path to the model. Defaults to ''.
        sample_name (str, optional): name of the sample. Defaults to 'uni_SC_nogo'.
        model_name (str, optional): name of the model. Defaults to 'g_d_b'.

    Returns:
        pyddm.model: loaded model
    """
    
    if file_path is None:
        file_path = construct_model_path(**kwargs)

    paramset_name = file_path.stem.split('params_')[-1]
    # 

    modelparams =  read_pickle(file_path)

    modelfun,full_params,hyperparams = get_model(which=model_class)  # get the model functions and parameters

    paramsets = get_delta_param_sets(full_params,which=model_class)  # get the reduced parameter sets

    model = pyddm.gddm(parameters=paramsets[paramset_name],**modelfun,**hyperparams)  # assemble the model with the parameters and functions
    model.set_model_parameters(modelparams['fitted'])
    return model



# %%
