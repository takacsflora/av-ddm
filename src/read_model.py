
#%%
from AVmodel import get_param_sets,assemble_model
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



def get_model(file_path = None,**kwargs): 
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

    model_name = file_path.stem.split('Model_')[-1]
    # 

    modelparams =  read_pickle(file_path)

    models = get_param_sets()
    model = assemble_model(models[model_name])
    model.set_model_parameters(modelparams['fitted'])
    return model



# %%
