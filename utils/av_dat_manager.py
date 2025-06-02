
from pathlib import Path
import pandas as pd 
import numpy as np


import pyddm 
from sklearn.model_selection import StratifiedShuffleSplit
from src.my_io import save_pickle,read_pickle
"""
functions to preprocess the data that is stored in pd.df extracted from the PinkRig pipeline

"""
def get_inactivation_locations():
    inactivation_locations = {
        'AV029': 'SC',
        'AV031': 'SC',
        'AV033': 'SC',
        'AV036': 'SC',
        'AV038': 'SC',
        'AV041': 'SC',
        'AV044': 'SC',
        'AV046': 'SC',
        'AV047': 'SC',
        'AV052': 'MOs',
        'AV053': 'MOs',
        'AV055': 'SC',
        'AV056': 'MOs',
        'AV057': 'MOs'
    }

    return inactivation_locations

def preproc_ev(ev): 
    """
    function to preprocess the event structure coming out of the PinkRig pipeline to be suitable for pyddm

    Parameters:
    -----------

    ev: pd.df  
        specifies whether to calculate a new column called trainSet for holding out data 
        split occurs based on sklearn.StratifiedShuffleSplit
        Stratification occurs based on visStim, audStim and choice
    splitkwargs: dict
        input parameters for  sklearn.StratifiedShuffleSplit 

    """
    ev['visDiff'] = ev.stim_visDiff
    ev['audDiff'] = ev.stim_audDiff
    ev['visDiff'] = np.round(ev.visDiff/max(ev.visDiff),2) # aud is already normalised byt we also normalise vis
    ev["choice"] = (ev["response_direction"]-1).astype(int)
    # here we can play around a bit... 
    stim_on  = ev[['timeline_audPeriodOn','timeline_visPeriodOn']].min(axis=1)
    ev['stim_on'] = stim_on
    ev['rt_thresh'] = ev.timeline_choiceThreshOn-ev.stim_on # stim_on
    ev['rt_laserThresh'] = ev.timeline_choiceThreshPostLaserOn-ev.block_laserStartTimes # block laserStartTimes   
    ev['laserstim_diff'] = ev.block_laserStartTimes-ev.stim_on 
    ev['stimulated_hemisphere'] = np.sign(ev.laser_power_signed)

    
    visContrast = np.abs(ev["visDiff"])
    visSide = np.sign(ev["visDiff"])
    audSide = np.sign(ev["audDiff"])
    isOpto = ev['is_laserTrial']
    
    #  variables that I actually use in the fitting..
    a_R = (audSide>0)
    a_L = (audSide<0)
    v_R = (visSide>0) * visContrast
    v_L = (visSide<0) * visContrast

    ev['isOpto'] = isOpto
    ev['a_R'] = a_R
    ev['a_L'] = a_L
    ev['v_R'] = v_R
    ev['v_L'] = v_L

    return ev

def filter_ev(ev,rt_rel_to = 'stim',ctrl_only = False):
    """
    function to filter the event structure coming out of the PinkRig pipeline
     - throws away nogo trials 
     - selects RT relative to stim or laser onset

    Parameters:
    -----------
    ev: pd.df  
    rt_rel_to: str
        specifies whether to use the RT relative to the stim or the laser onset. 

    """
    
    if rt_rel_to == 'stim':
        ev['RT'] = ev['rt_thresh']
    elif rt_rel_to == 'laser':
        ev['RT'] = ev['rt_laserThresh']    
    ev = ev[((ev.choice==0) | (ev.choice==1)) & 
            (~ev.RT.isna())].copy()

    if ctrl_only:
        ev = ev[ev['is_laserTrial'] == 0].copy()

    return ev

def read_csvs(set_name = 'uni_SC_nogo', subset =''):
    """read all the csvs that are saved out fro mthe pinkRig pipeline as a folder and concatenate into a single df. Works by glob gob search

    Args:
        set_name (str, optional): the folder identifier eg. uni_SC. Defaults to 'uni_SC_nogo'.
        subset (str, optional): can call string identifiers e.g.just a subset of files within that folder. For example AV036. Defaults to ''.

    Returns:
        pd.df: all the csvs concatenated into a single df
    """
    
    main_path  = Path(rf'D:\LogRegression\opto\{set_name}')
    files = list(main_path.glob(f'{subset}*.csv'))
    data = []
    for path in files:
        df = pd.read_csv(path)
        df = preproc_ev(df.copy())
        df['file'] = path.stem
        df['subject'] = df['file'].str.split('_').str[0]

        data.append(df)


    return pd.concat(data, ignore_index=True)

def get_summary_dataset(set_name='uni_SC_nogo',recompute=True,subsample=True,rt_rel_to = 'stim'):
    """function to save and extract the summary data from a given dataset

    Args:
        dataset (str, optional): identifier of the dataset. Defaults to 'uni_SC_nogo'.
        recompute (bool, optional): whether the reextract he sample. Defaults to True.
        subsample (bool, optional): each file will have he same number of trials in the summary data if set True. Defaults to True.

    Returns:
        pyddm.sample: to be used for fitting the model
    """
    savepath = Path(rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_{rt_rel_to}\summary_data')
    file_name = f'{set_name}.pickle'
    file_path = savepath / file_name

    if not file_path.exists() or recompute:
        savepath.mkdir(parents=True,exist_ok=True)
        df = read_csvs(set_name)
        df = preproc_ev(df)
        df = filter_ev(df,rt_rel_to = rt_rel_to)

        if subsample: 
            min_trials_per_file = df['file'].value_counts().min()
            df = df.groupby('file', group_keys=False).apply(lambda x: x.sample(n=min_trials_per_file, random_state=42))

        sample = pyddm.Sample.from_pandas_dataframe(df, 
                                                    rt_column_name="RT", 
                                                    choice_column_name="choice", 
                                                    choice_names =  ("Right", "Left"))



        save_pickle(sample,file_path)
    else:
        sample = read_pickle(file_path)

    return sample 

def cv_split(ev,**splitkwargs):
    """
    function to calculate a new column called trainSet for holding out data 
    split occurs based on sklearn.StratifiedShuffleSplit
    Stratification occurs based on visStim, audStim and choice

    Parameters: 
    -----------
    ev: pd.df  
    splitkwargs: dict
        input parameters for  sklearn.StratifiedShuffleSplit 
        
    """

    _,ev['groupID'], counts = np.unique(ev[['visDiff','audDiff','choice']],axis=0,return_inverse=True,return_counts=True)

    # in some datasets the count is 1 for certain groups -- that is not worth estimating so we throw this datapoint out
    singles = np.where(counts==1)[0]

    if singles.size>0:
        is_single_count = np.sum(np.concatenate([(ev.groupID==smallset).values.astype('int')[np.newaxis,:] for smallset in np.where(counts==1)[0]]),axis=0)
        is_single_count = is_single_count.astype('bool')
        ev=ev[~is_single_count]

    sss = StratifiedShuffleSplit(**splitkwargs)
    _,(train_idx,_) = sss.split(ev['RT'],ev[['choice','groupID']])
    
    
    ev = ev.copy()  # Ensure we are working on a copy to avoid SettingWithCopyWarning
    ev['trainSet'] = False
    ev.loc[ev.index[train_idx], 'trainSet'] = True


    return ev

def resample_model(model,sample_path=None,split=True):
    """
    resampling of a pyddm model for model recovery (i.e. generate new samples andrefit the model to see how reliably we can distinguish this rom other models)

    # we just need to read in how many choices were made in the actial sample with that condition for model recovery. 
    """

    if sample_path: 
        data = read_pickle(sample_path)
        actual_aud_azimuths = np.sort(data.to_pandas_dataframe().audDiff.unique())
        actual_vis_contrasts = np.sort(data.to_pandas_dataframe().visDiff.unique())
    else:
        actual_aud_azimuths = [-60,0,60]
        actual_vis_contrasts = [-1,-.5,-.25,0,.25,.5,1]
    sample_df = []

    for isLaser in range(2):
        for ia,a in enumerate(actual_aud_azimuths):
            for i,v in enumerate(actual_vis_contrasts):
                curr_cond = {'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
                sol = model.solve(conditions=curr_cond)

                if sample_path: 
                    d_set = data.subset(audDiff=a,visDiff=v,is_laserTrial=isLaser)
                    n_samples = d_set.choice_upper.size  + d_set.choice_lower.size
                else: 
                    n_samples = 50

                sample = sol.resample(n_samples)
                sample_df.append(sample.to_pandas_dataframe(drop_undecided=True))

    sample_df = pd.concat(sample_df)

    if split:
        Block = cv_split(sample_df,n_splits=2,test_size=.2,random_state=0)
        train = pyddm.Sample.from_pandas_dataframe(Block[Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        test = pyddm.Sample.from_pandas_dataframe(Block[~Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
    else: 
        
        train = pyddm.Sample.from_pandas_dataframe(sample_df,
            rt_column_name='RT',
            choice_column_name='choice',
            choice_names =  ("Right", "Left"))
        test=None

    
    return train,test

def write_samples():
    """function to write pyddm samples from csvs in a given folder. Writes a sample individually for all csvs and also splots into train and test sets.
    Output will create 3 folders of sample files (pickle) in the savepath directory.
    train, test and all.
    """

    data_path = Path(r'D:\LogRegression\opto\uni_MOs_nogo')
    animal_paths = list(data_path.glob('*.csv'))
    rt_rel_to = 'stim' # 'stim' or 'laser'

    savepath = Path(rf'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_{rt_rel_to}')

    ctrl_only = False

    if ctrl_only:
        savepath = savepath #/ 'ctrl'
    # output data paths
    savetrain = savepath / 'train'
    savetest = savepath / 'test'
    saveall = savepath / 'all'

    savetrain.mkdir(parents=True,exist_ok=True)
    savetest.mkdir(parents=True,exist_ok=True)
    saveall.mkdir(parents=True,exist_ok=True)

    for animal_path in animal_paths:

        ev = pd.read_csv(animal_path) 
        ev = preproc_ev(ev)

        s = animal_path.stem
        Block = filter_ev(ev,rt_rel_to = rt_rel_to,ctrl_only=ctrl_only)

        if Block.shape[0] == 0:
            print('no data for %s' % s)
            continue
        else: 
            print('processing %s' % s)

            Block["choice"] = Block["choice"].to_numpy()
            Sample_ = pyddm.Sample.from_pandas_dataframe(Block, rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
            save_pickle(Sample_,saveall / ('%s_Sample_all.pickle' % s))

            Block = cv_split(Block,n_splits=2,test_size=.2,random_state=0)
            Sample_train = pyddm.Sample.from_pandas_dataframe(Block[Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
            save_pickle(Sample_train,savetrain / ('%s_Sample_train.pickle' % s))
            Sample_test = pyddm.Sample.from_pandas_dataframe(Block[~Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
            save_pickle(Sample_test,savetest / ('%s_Sample_test.pickle' % s))


if __name__ == "__main__":  
   write_samples() 
