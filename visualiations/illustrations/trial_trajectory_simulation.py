#%%

import numpy as np
import matplotlib.pyplot as plt


SAVE_PATH = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots'


plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})


def lazy_noise(lazy_rate,t):
    return 1/np.exp(lazy_rate*t)

# Or, alternatively, the vectorized four-line version
def run_vanilla_ddm(drift_rate, noise=1, bound=1, T_dur=1.5, dt=.01,lazy_rate=None):
    # Run the actual trajectory
    if lazy_rate is not None:
        t = np.arange(0,T_dur,dt)
        noise = lazy_noise(lazy_rate,t)
        drift_rate = drift_rate * noise

    trajectory = np.cumsum(drift_rate*dt + noise*np.sqrt(dt)*np.random.randn(int(T_dur/dt)))
    # Find the first place where it crosses the upper or lower bound.  If there
    # was no bound crossing, consider the RT to be the end time (forcing a
    # choice).
    RT_index = min(np.where(np.abs(trajectory)>bound)[0], default=len(trajectory)-1)
    # Calculate the RT from the bound crossing
    RT = RT_index * dt
    # Check whether it is the upper or lower bound we crossed or none
    if trajectory[RT_index] > bound:
        choice = 'Right'
    elif trajectory[RT_index] < -bound:
        choice = 'Left'
    else:
        choice = 'NoGo'
    #choice = int(trajectory[RT_index] > 0)
    return trajectory[0:RT_index], choice, RT




#%%

dt = 0.01
T_dur = 1.5
lazy_rate = 1.5
drift_rate = 0
t = np.arange(0,T_dur,dt)

noise = lazy_noise(lazy_rate,t)
drift_rate = drift_rate * noise

# drift_rate = 0
# noise = np.ones_like(t) * 0.3
dx = drift_rate*dt + noise*np.sqrt(dt)
trajectory = np.cumsum(dx)


fig,ax = plt.subplots(2,1,figsize=(.5,1),dpi=150,sharex=True,sharey=True)
ax[1].plot(t, dx,'k-')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('dx')


drift_rate = 0
noise = np.ones_like(t) * 1
dx = drift_rate*dt + noise*np.sqrt(dt)
ax[0].plot(t, dx,'k-')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('dx')

ax[0].set_ylim([0,0.15])

plt.savefig(f'{SAVE_PATH}/ddm_dx.svg',dpi=300,bbox_inches='tight')

#%%
# %%
hyperparams ={ 
    'dt':.01,
    'T_dur' : 1.5,
    'bound':1, 
    'lazy_rate': None #1.5
}


choice_palette = {'NoGo': "#C87801", 'Left': '#FF00FF', 'Right': "#07E603"}



fig, ax = plt.subplots(1,1,figsize=(1,1),dpi=150)




bounary_pkws = {'linestyle':':','color':'gray','alpha':0.9}

x0_line = np.arange(-.2, hyperparams['T_dur'], hyperparams['dt'])
t_full = np.arange(0, hyperparams['T_dur'], hyperparams['dt'])


ax.plot(t_full, np.ones_like(t_full)*hyperparams['bound'],**bounary_pkws)
ax.plot(t_full, -np.ones_like(t_full)*hyperparams['bound'],**bounary_pkws)
ax.plot(x0_line, np.zeros_like(x0_line),**bounary_pkws)
ax.plot([0,0],[-hyperparams['bound'],hyperparams['bound']],**bounary_pkws)

ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])


n_trials = 50
alpha= 0.1
for i in range(n_trials):
    dx_t, choice, rt = run_vanilla_ddm(drift_rate=0, noise=.3,**hyperparams)
    t_trial = np.arange(0.001, rt, hyperparams['dt'])
    ax.plot(t_trial, dx_t, color = choice_palette[choice],alpha=alpha)

for i in range(n_trials):
    dx_t, choice, rt = run_vanilla_ddm(drift_rate=1.5, noise=.3,**hyperparams)
    t_trial = np.arange(0.001, rt, hyperparams['dt'])
    ax.plot(t_trial, dx_t, color = choice_palette[choice],alpha=alpha)

for i in range(n_trials):
    dx_t, choice, rt = run_vanilla_ddm(drift_rate=-1.5, noise=.3,**hyperparams)
    t_trial = np.arange(0.001, rt, hyperparams['dt'])
    ax.plot(t_trial, dx_t, color = choice_palette[choice],alpha=alpha)

# dx_t, choice, rt = run_vanilla_ddm(drift_rate=0, noise=.3,**hyperparams)
# t_trial = np.arange(0.001, rt, hyperparams['dt'])
# ax.plot(t_trial, dx_t, color = choice_palette[choice],alpha=1)


plt.savefig(f'{SAVE_PATH}/ddm_ttrajectories_uniform.svg',dpi=300,bbox_inches='tight')


# for i in range(50):
#     dx_t, choice, rt = run_vanilla_ddm(drift_rate=0, noise=.7,**hyperparams)
#     t_trial = np.arange(0.001, rt, hyperparams['dt'])
#     ax.plot(t_trial, dx_t, color = choice_palette[choice],alpha=0.1)



# %%

