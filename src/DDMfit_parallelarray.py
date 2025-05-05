import pyddm 
import time
import itertools
import os 
import psutil
import sys

from fitting import get_parameters
from model_components import get_freeP_ctrl,get_freeP_opto,get_freeP_set
from my_io import save_pickle,read_pickle
from pathlib import Path

# for the parallel 
print('CSOLVE CORRECRLY LOADED:', pyddm.model.HAS_CSOLVE)

def trainDDMs(rank=1):
    """
    primarily a parallelisable finction to fit ddms
    
    """
    rank = float(rank)
    #pyddm.set_N_cpus(16) # set to 6 core/cpu
    # input data paths

    fit_resampled_data = False 
    fit_type = 'opto'

    mycwd = Path(os.getcwd())
    
    if fit_resampled_data:
         data_path = mycwd / 'resample_data/train'

    else:
        data_path = mycwd / 'summary_data'
    
    #data_path = Path(r'C:\Users\Flora\Documents\Github\av-ddm\data\rt_to_stim\summary_data')

    animal_paths = list(data_path.glob('*.pickle')) # 11 files 



    if fit_resampled_data:
        drift_class = 'resampled_fits'
    else:
        drift_class = 'model_fits'

    savepath = mycwd / drift_class
    savepath.mkdir(parents=True,exist_ok=True)


    freeP_sets = get_freeP_set(fit_type=fit_type) # the differnt model types we are testing

    print(len(animal_paths)*len(freeP_sets), 'models to fit...')

    for i,(animal_path,set) in enumerate(itertools.product(animal_paths,freeP_sets)):
        s = animal_path.stem
        currmodel_path  = savepath / ('%s_Model_%s.pickle' % (s,set)) 
        if (i==(rank-1)) and not currmodel_path.is_file(): 
            # preproc
            Sample_train = read_pickle(animal_path) 

            t0 = time.time()
            
            if fit_type == 'ctrl': 
                freePs = get_freeP_ctrl(set)
            elif fit_type == 'opto':
                freePs = get_freeP_opto(set)
            
            fit_params = get_parameters(fit_type=fit_type,freePs=freePs) 


            m = pyddm.Model(**fit_params)
            pyddm.fit_adjust_model(model=m, sample=Sample_train, lossfunction=pyddm.LossLikelihood, verbose=False)         
            save_pickle(m,savepath / currmodel_path)
            print('%s fit %s...' %(set,s))
            print('time to fit:%.2d s' % (time.time()-t0))    
            print('RAM Used (GB):', (psutil.virtual_memory()[3]/1000000000))
            print('memory used (GB):', (psutil.Process().memory_info().rss / (1024 * 1024 *1000)))
            
if __name__ == "__main__":  
   trainDDMs(rank=sys.argv[1]) 
   #trainDDMs(rank=1) 
