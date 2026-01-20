
from pathlib import Path
import pandas as pd
import numpy as np

import re
import pyddm 

from my_io import read_pickle
from read_model import load_saved_model
from undecided import LossLikelihoodUndecided

from utils.av_dat_manager import get_inactivation_locations
inactivation_locations = get_inactivation_locations()

def get_dataset(basepath = None):
    
    if basepath is None:
        basepath = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_laser')

    sample_path_train = basepath / 'train'
    sample_path_test = basepath / 'test'
    model_path = sample_path_train / 'model_fits'


    datasets = pd.DataFrame({'train_path':list(sample_path_train.glob('*.pickle'))})

    datasets['stem'] = datasets['train_path'].apply(
        lambda path: pd.Series(path.stem.split('_Sample_train')[0])
    ) 

    datasets[['subject', 'hemisphere', 'power']] = datasets['stem'].apply(
        lambda stem: pd.Series(stem.split('_')[:3] + [None] * (3 - len(stem.split('_'))))
    )

    # Map inactivation locations to datasets based on the subject
    datasets['inactivation_location'] = datasets['subject'].map(inactivation_locations)
    # find the corresponding test set for each dataset
    datasets['test_path'] = datasets['stem'].apply(
        lambda stem: sample_path_test / ('%s_Sample_test.pickle' % (stem))
    )

    # Find all model paths for each training set
    datasets['model_paths'] = datasets['stem'].apply(
        lambda stem: list(model_path.glob(f'{stem}_*.pickle'))
    )
    # Extract model names from model paths
    datasets['model_types'] = datasets['model_paths'].apply(
        lambda paths: [re.search(r'_model_(.*?)_params_', path.stem).group(1) for path in paths]
    )
    

    datasets['params_tested'] = datasets['model_paths'].apply(
        lambda paths: [re.split(r'params_', path.stem, maxsplit=1)[-1] for path in paths]
    )
    

    datasets['n_models'] = datasets['model_paths'].apply(len)

    return datasets

def get_model_parameters(model):
    """
    local helper to get the parameters of a model
    Parameters: 
    model: pyddm.Model 
    """
    # get the parameters of the model
    all_params = {
        param_subset: model.parameters()[param][param_subset].real
        for param in model.parameters()
        for param_subset in model.parameters()[param]
    }
    
    return pd.DataFrame([all_params])

def get_logliks(sample):
    """
    local helper to get the LogLik of a model on the test set and the training set
    Parameters: 
    dataset: pd.Series 
    """
    # Load the test sample
    test_sample = read_pickle(sample['test_path'])
    train_sample = read_pickle(sample['train_path'])
    
    # Load the model
    model_paths = sample['model_paths']
    model_types = sample['model_types']
    paramset = sample['params_tested']

    LogLiks_train,LogLiks_test = [],[]
    LogLiks_train_undec,LogLiks_test_undec = [],[]

    params = []
    for path, model_type, param in zip(model_paths, model_types, paramset): 
        print(path.stem)
        model = load_saved_model(path,model_class=model_type)
        n_trials_train = train_sample.choice_upper.size + train_sample.choice_lower.size
        n_trials_test = test_sample.choice_upper.size + test_sample.choice_lower.size

        LogLiks_train.append(pyddm.get_model_loss(model=model,sample=train_sample,lossfunction=LossLikelihoodUndecided)/(n_trials_train))
        LogLiks_test.append(pyddm.get_model_loss(model=model,sample=test_sample,lossfunction=LossLikelihoodUndecided)/(n_trials_test))


        # LogLiks_train_undec.append(pyddm.get_model_loss(model=model,sample=train_sample,lossfunction=LossLikelihoodUndecided)/(train_sample.undecided))
        # LogLiks_test_undec.append(pyddm.get_model_loss(model=model,sample=test_sample,lossfunction=LossLikelihoodUndecided)/(test_sample.undecided))

        params.append(get_model_parameters(model))
    
    model_params = pd.concat(params,ignore_index=True)


    # normalise the LogLiks between control and full model



    results = pd.DataFrame({
        'model_type': model_types,
        'paramset': paramset,
        'LogLik_train': LogLiks_train,
        'LogLik_test': LogLiks_test,
        # 'LogLik_train_undec': LogLiks_train_undec,
        # 'LogLik_test_undec': LogLiks_test_undec,
        'subject': sample['subject'],
        'hemisphere': sample['hemisphere'],
        'inactivation_location': sample['inactivation_location'],
        'power': sample['power'],
        'stem': sample['stem']
    })

    logLik_cols = ['LogLik_train','LogLik_test'] #,'LogLik_train_undec','LogLik_test_undec']
    # check wheter paramsets contain ctrl and full
    if (results.paramset=='full').any() and (results.paramset=='ctrl').any():
        to_normalise = True
    else:
        to_normalise = False
    
    if to_normalise:
        for logLik_col_name in logLik_cols:
            full = results[results['paramset'] == 'full'][logLik_col_name].values[0]
            ctrl = results[results['paramset'] == 'ctrl'][logLik_col_name].values[0]

            delta_loglik = (results[logLik_col_name] - ctrl)/(full-ctrl)
            results[f'{logLik_col_name}_norm'] = delta_loglik

    



    return pd.concat([results, model_params], axis=1)

def load_evaluation(path = None,recompute = False):
    """
    function to run the evaluation of the model on the test set and the training set
    """
    if path is None:
        path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim')

    eval_path = path / 'model_evaluation.csv'
    if eval_path.exists() and not recompute:
        print('Looading evaluation...')        
    else:
        print('Recomputing evaluation...')

        datasets = get_dataset(path)
        datasets = datasets[datasets['n_models'] > 0].reset_index(drop=True)

        # sometimes one or two model doesn't fit...
        total_model_no = datasets.n_models.max()
        datasets = datasets[datasets['n_models'] == total_model_no].reset_index(drop=True)  # temporary hack because AV036 didnot fit

        # removing the summary model from evaluation (we evaluate per subject)
        datasets = datasets[~datasets['subject'].isin(['summary'])].reset_index(drop=True)


        results = [get_logliks(dataset) for _, dataset in datasets.iterrows()]
        results = pd.concat(results, axis=0).reset_index()

        results.to_csv(eval_path, index=False)

    return pd.read_csv(eval_path)

def get_gain_loss_models(df):
    """
    helper function to get the gain and the loss models for a given parameter 
    """
    unique_models = df['paramset'].unique()

    # Organize the data for gain and loss models
    # Extract unique stems from the 'stem' column in the dataframe
    unique_stems = df['stem'].unique()

    # Initialize a list to store the results
    rows = []


    # Iterate over each stem and parameter
    LogLik_cols = ['LogLik_train', 'LogLik_test', 'LogLik_train_norm', 'LogLik_test_norm',
                    #'LogLik_train_undec', 'LogLik_test_undec', 'LogLik_train_undec_norm', 'LogLik_test_undec_norm',
                'subject', 'hemisphere', 'inactivation_location',
                                    ]
    for stem in unique_stems:
        stem_df = df[df['stem'] == stem]
        for model in unique_models:
            if model.startswith('g_') or model.startswith('l_'):
                param = model[2:]  # Extract parameter name
                row = {'stem': stem, 'param': param}
                if model.startswith('g_'):
                    row.update({'type': 'gain'})
                    row.update(stem_df[stem_df['paramset'] == model][LogLik_cols].iloc[0].to_dict())
                elif model.startswith('l_'):
                    row.update({'type': 'loss'})
                    row.update(stem_df[stem_df['paramset'] == model][LogLik_cols].iloc[0].to_dict())
                rows.append(row)
            else: 
                param = 'ctrl'
                row = {'stem': stem, 'param': param}
                if model.startswith('ctrl'):
                    row.update({'type': 'gain'})
                    row.update(stem_df[stem_df['paramset'] == model][LogLik_cols].iloc[0].to_dict())
                elif model.startswith('full'):
                    row.update({'type': 'loss'})
                    row.update(stem_df[stem_df['paramset'] == model][LogLik_cols].iloc[0].to_dict())
                rows.append(row)

    # Convert he list of rows into a pandas DataFrame
    results_df = pd.DataFrame(rows)

    return results_df