
#%%

import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt




left_bound =1.5 
right_bound = 1
starting_point = 0
slope = 0.5


def create_drift_arrow(starting_point,slope,stim_at=200,**kwargs):

    start = (stim_at, starting_point)
    end = (stim_at+400, starting_point+slope)

    # Create the arrow
    arrow = FancyArrowPatch(start, end, 
                            arrowstyle='-|>',  # nice arrowhead
                            mutation_scale=20,**kwargs)
    return arrow

def drift_illustration(ax, starting_point=0, slope=0.5, left_bound=1, right_bound=1,add_arrow=True,arrow_linestyle='--'):
    n_samples = 1000
    time = np.arange(n_samples)   
    stim_at = 200


    zero_line = np.zeros(n_samples)
    left_bound_line = np.concatenate((np.zeros(stim_at),np.ones(n_samples-stim_at)*left_bound))
    right_bound_line = np.concatenate((np.zeros(stim_at),np.ones(n_samples-stim_at)*right_bound))
    #drift_vector = 

    ax.plot(time,zero_line,linestyle='--',color='grey',linewidth=1)
    ax.plot(time,right_bound_line,linestyle='-',color='grey',linewidth=1)
    ax.plot(time,-left_bound_line,linestyle='-',color='grey',linewidth=1)

    # Create a drift vector with a slope
    # Add an arrow to represent the drift vector
    if add_arrow:
        arrow = create_drift_arrow(starting_point,slope,stim_at=stim_at, color='purple',linestyle=arrow_linestyle,linewidth=2)
        ax.add_patch(arrow)


#%%

fig,a = plt.subplots(1,1,figsize=(4,3),sharex=True,sharey=True)
drift_illustration(a,starting_point=0,slope=0.5,left_bound=1,right_bound=1,add_arrow=False)
arrow= create_drift_arrow(starting_point=0,slope=0.5,stim_at=200, color='k',linestyle='-',linewidth=2)
a.add_patch(arrow)
a.set_xticks([])
a.set_yticks([])
for spine in a.spines.values():
    spine.set_visible(False)
# %%

fig,ax = plt.subplots(5,2,figsize=(3,10),sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.01,hspace=0.01)

# ctrl 
def plot_control(axs,**kws):
    ax0,ax1 = axs
    drift_illustration(ax0,starting_point=0,slope=0.3,**kws)
    drift_illustration(ax1,starting_point=0,slope=0.5,**kws)

opto_arrow_kws = {'color':'orange','linestyle':'-','linewidth':2}


plot_control(ax[0,:], arrow_linestyle='-')

plot_control(ax[1,:])
arrow0 = create_drift_arrow(starting_point=.5,slope=0.3,**opto_arrow_kws)
ax[1,0].add_patch(arrow0)
arrow1 = create_drift_arrow(starting_point=.5,slope=0.5,**opto_arrow_kws)
ax[1,1].add_patch(arrow1)


plot_control(ax[2,:])
arrow0 = create_drift_arrow(starting_point=0,slope=0.6,**opto_arrow_kws)
ax[2,0].add_patch(arrow0)
arrow1 = create_drift_arrow(starting_point=0,slope=0.8,**opto_arrow_kws)
ax[2,1].add_patch(arrow1)

plot_control(ax[3,:])
arrow0 = create_drift_arrow(starting_point=0,slope=0.3,**opto_arrow_kws)
ax[3,0].add_patch(arrow0)
arrow1 = create_drift_arrow(starting_point=0,slope=0.8,**opto_arrow_kws)
ax[3,1].add_patch(arrow1)


plot_control(ax[4,:])
ax[4,0].plot(np.arange(200,1000,1),np.ones(800)*-1.3,**opto_arrow_kws)
ax[4,1].plot(np.arange(200,1000,1),np.ones(800)*-1.3,**opto_arrow_kws)





for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
for a in ax.flatten():
    for spine in a.spines.values():
        spine.set_visible(False)
# %%
