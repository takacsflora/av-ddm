# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

import plots 
from src.my_io import read_pickle




refit_options = [
    'all',
    'l_aS',
    'l_vS',
    'l_d_mixturecoef',
    'g_d_b',
    'g_both', 
    'g_boundx0',
    'g_d_x0',
    'l_v', 
    ]

#for subject in subjects:

type = refit_options[4]
plot_log = True
to_save = True

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')

#
subjects = list((basepath/'ClusterResults/DriftAdditiveOpto').glob('*Model_all.pickle'))
subjects = [re.findall(r'(\w+)_Sample_train_Model_all', s.stem)[0] for s in subjects]
#subjects = ['AV047_10mW_Left']
for subject in subjects:
    model_path  = basepath / 'ClusterResults/DriftAdditiveOpto'

    #model_path = Path(r'C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora\behavior\opto\ddm\DriftAdditiveOpto')

    sample_path = basepath / 'Data/forMyriad/samples/train'

    sample_path = (sample_path / ('%s_Sample_train.pickle' % (subject)))


    model_path = (model_path / ('%s_Model_%s.pickle' % (sample_path.stem,type)))


    # data_path = basepath / ('Data/%s.csv' % subject)
    # ev = pd.read_csv(data_path) 
    # ev = preproc_ev(ev)
    # Block = ev[~np.isnan(ev.rt_laserThresh) & ~ev.is_laserTrial.astype('bool')] 

    model = read_pickle(model_path)
    sample = read_pickle(sample_path)
    model.parameters()
    #
    print('LogLik',type,model.fitresult.value())
    fig = plots.av_diagnostics(sample,model=model)
    fig.suptitle('%s_%s'% (subject,type))

    mypath = r'D:\Illustrations\Takacs et al paper'
    savename = mypath + '\\' + 'diagnostics_%s_%s.svg' % (subject,type)
    fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# #ax[0].set_yscale('symlog')
# %% PSYCHOMETRIC

fig,ap = plt.subplots(1,2,figsize=(6,3),sharey=True)
plots.plot_psychometric(model,sample,axctrl=ap[0],axopto=ap[1],plot_log=True)
fig.suptitle('psychometric')

savename = mypath + '\\' + 'ps_%s.svg' % subject
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
fig,ac = plt.subplots(1,2,figsize=(6,3),sharey=True)
wh = 'all'
plots.plot_chronometric(model,sample,axctrl=ac[0],axopto=ac[1],which=wh,metric_type='median')


savename = mypath + '\\' + 'chrono_%s.svg' % subject
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
fig.suptitle('%s choices' % wh)

# fig.suptitle(refitted)

# %%
