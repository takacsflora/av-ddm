
#%%
from .my_io import read_pickle
from pathlib import Path

def get_model(path = '', sample_name = 'uni_SC_nogo',model_name = 'g_d_b'): 
    """
    function to load a model from a given path. 

    Args:
        path (str, optional): path to the model. Defaults to ''.
        sample_name (str, optional): name of the sample. Defaults to 'uni_SC_nogo'.
        model_name (str, optional): name of the model. Defaults to 'g_d_b'.

    Returns:
        pyddm.model: loaded model
    """
    
    if path == '':
        path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\summary_data')
    else:
        path = Path(path)

    path = path / 'model_fits'
    
    file_path = path / f'{sample_name}_Model_{model_name}.pickle'
    return read_pickle(file_path)



# %%
