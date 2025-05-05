

import numpy as np
import matplotlib.pyplot as plt
import itertools
from pyddm.functions import solve_partial_conditions


def plot_diagnostics(model=None,sample = None, conditions=None,data_dt =.025,method=None,myloc=0,ax = None,dkwargs = None,mkwargs =None,time_on_x = True,data_plot_func='fill_between'):
    """
    visually assess the diagnostics of the model fit

    """
    if not ax:
        _,ax = plt.subplots(1,1)

    if not dkwargs:
        dkwargs = {
            'alpha' : .5, 
            'color' : 'k'
        }

    if not mkwargs: 
        mkwargs = {
            'lw': 2, 
            'color': 'k'
        }

    if model:
        T_dur = model.T_dur
        if model.dt > data_dt:
            data_dt = model.dt
    elif sample:
        T_dur = max(sample)
    else:
        raise ValueError("Must specify non-empty model or sample in arguments")

    # If a sample is given, plot it behind the model.
    if sample:
        s = sample.subset(**conditions)
        t_domain_data = np.linspace(0, T_dur, int(T_dur/data_dt+1))
        data_hist_top = np.histogram(s.choice_upper, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        data_hist_bot = np.histogram(s.choice_lower, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        total_samples = len(s)
        
        data_top_norm = np.asarray(data_hist_top)/total_samples/data_dt+myloc
        data_bot_norm = -np.asarray(data_hist_bot)/total_samples/data_dt+myloc
        datfunc = getattr(ax, data_plot_func)
        if time_on_x:
            datfunc(t_domain_data,data_top_norm, label="Data", **dkwargs)
            datfunc(t_domain_data+myloc,data_bot_norm, label="Data", **dkwargs)
        else:
            datfunc(data_top_norm,t_domain_data, label="Data",**dkwargs)
            datfunc(data_bot_norm,t_domain_data, label="Data", **dkwargs)
        toplabel,bottomlabel = sample.choice_names
    if model:
        s = solve_partial_conditions(model, sample, conditions=conditions, method=method)

        if time_on_x:
            ax.plot(model.t_domain(),s.pdf("_top"),**mkwargs)
            ax.plot(model.t_domain(),-s.pdf("_bottom"),**mkwargs)

        else:
            ax.plot(s.pdf("_top")+myloc,model.t_domain(),**mkwargs)
            ax.plot(-s.pdf("_bottom")+myloc,model.t_domain(),**mkwargs)

        toplabel,bottomlabel = model.choice_names

def get_rt_quartiles(m,a,v,o,which = 'correct'):
    sol = m.solve(conditions={"audDiff": a, "visDiff": v,'is_laserTrial':o})

    if 'correct' in which:
        side = np.sign(np.sign(a) + np.sign(v)) 
    elif 'left' in which: 
        side = -1
    elif 'right' in which:
        side = 1
    else:
        side = 0 # when we want to get both

    percentiles = [.45,.5,.55]
    l = [np.interp(np.ptp(sol.cdf('Left'))*p,sol.cdf('Left'),sol.t_domain) for p in percentiles]
    r = [np.interp(np.ptp(sol.cdf('Right'))*p,sol.cdf('Right'),sol.t_domain) for p in percentiles]
    j = [np.interp(np.ptp(sol.cdf('Right')+sol.cdf('Left'))*p,sol.cdf('Right')+sol.cdf('Left'),sol.t_domain) for p in percentiles]

    if side==-1: 
        lower,mid,upper = l
    elif side==1: 
        lower,mid,upper = r
    elif side==0: 
        lower,mid,upper = j

    return lower,mid,upper 

def get_median_rt(sample,a,v,o,min_N=10,which='correct',metric_type = 'median'):
    dat = sample.subset(audDiff=a,visDiff=v,is_laserTrial=o)
    l = dat.choice_lower
    r = dat.choice_upper 

    if 'correct' in which:
        side = np.sign(np.sign(a) + np.sign(v)) 
    elif 'left' in which: 
        side = -1
    elif 'right' in which:
        side = 1
    else:
        side = 0 # when we want to get both

    if side==-1: 
        if l.size>min_N:
            rt_ = l
        else:
            rt_ = None

    elif side==1:
        if r.size>min_N: 
            rt_ = r
        else:
            rt_ = None 
    elif side==0: 
        j = np.concatenate((l,r))
        if j.size>min_N: 
            rt_ = j
        else:
            rt_ = None

    if rt_ is not None: 
        if 'mean' in metric_type:
            rt_ = np.mean(rt_)
        elif 'median' in metric_type: 
            rt_ = np.median(rt_)
    return rt_

def plot_psychometric(model,sample,axctrl=None,axopto=None,plot_log=False): 

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    aud_azimuths  = np.linspace(-1,1,3)
    vis_contrasts = np.linspace(-1,1,40)

    linestyles= ['--','-']
    markerstyles = [None,'filled']
    if axctrl is None or axopto is None:         
        _,(axctrl,axopto) = plt.subplots(2,1,figsize=(20,10))
    axes = [axctrl,axopto]

    for isLaser,(line,marker,ax) in enumerate(zip(linestyles,markerstyles,axes)):
        psychometric,a,v = zip(*[[model.solve(conditions={"audDiff": a, "visDiff": v,'is_laserTrial':isLaser}).prob('Right'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])
        psychometric = np.reshape(np.array(psychometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
        a = np.reshape(np.array(a),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
        v = np.reshape(np.array(v),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
        psychometric_actual = [sample.subset(audDiff=a,visDiff=v,is_laserTrial=isLaser).prob('Right') for a,v in itertools.product(actual_aud_azimuths,actual_vis_contrasts)]
        psychometric_actual = np.reshape(np.array(psychometric_actual),(actual_aud_azimuths.size,actual_vis_contrasts.size)) 
        psychometric_log = np.log10(psychometric/(1-psychometric))
        #mye = 1e-10
        psychometric_actual_log = np.log10((psychometric_actual+1e-4)/(1-(psychometric_actual)+1e-4))


        colors = ['b','k','r']

        if marker is None: 
            facecolors = ['None','None','None']
        else: 
            facecolors = colors

        if plot_log:
            gamma = model.parameters()['drift']['gamma'].real
            [ax.plot(np.abs(vis_contrasts)**gamma * np.sign(vis_contrasts),p,color=c,linestyle=line) for p,c in zip(psychometric_log,colors)]
            [ax.scatter(np.abs(actual_vis_contrasts)**gamma * np.sign(actual_vis_contrasts),p,color=c,marker='o',facecolors=fc) for p,c,fc in zip(psychometric_actual_log,colors,facecolors)]
            ax.set_ylim([-3,3])
            ax.axhline(0,color='k',linestyle='--')

        else:
            [ax.plot(vis_contrasts,p,color=c,linestyle=line) for p,c in zip(psychometric,colors)]
            [ax.scatter(actual_vis_contrasts,p,color=c,marker='o',facecolors=fc) for p,c,fc in zip(psychometric_actual,colors,facecolors)]
            ax.set_ylim([-.05,1.05])
            ax.axhline(0.5,color='k',linestyle='--')

        ax.axvline(0,color='k',linestyle='--')
        ax.set_ylabel('p(R)')
        ax.set_xlabel('contrasts')
    
    axctrl.set_title('control')
    axopto.set_title('opto')

def plot_chronometric(model,sample,which='correct',axctrl=None,axopto=None,metric_type='median'):

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    aud_azimuths  = np.linspace(-1,1,3)
    vis_contrasts = np.linspace(-1,1,40)
    colors = ['b','k','r'] # for -1,0,1 aud

    linestyles= ['--','-']
    markerstyles = [None,'filled']

    if axctrl is None or axopto is None:         
        _,(axctrl,axopto) = plt.subplots(2,1,figsize=(20,10))
    axes = [axctrl,axopto]

    for isLaser,(line,marker,ax) in enumerate(zip(linestyles,markerstyles,axes)):

        c_l,c_m,c_u = zip(*[get_rt_quartiles(model,a,v,isLaser,which=which) for a,v in itertools.product(aud_azimuths,vis_contrasts)])
        c_l = np.reshape(np.array(c_l),(aud_azimuths.size,vis_contrasts.size)) 
        c_m = np.reshape(np.array(c_m),(aud_azimuths.size,vis_contrasts.size)) 
        c_u = np.reshape(np.array(c_u),(aud_azimuths.size,vis_contrasts.size)) 

    # or just this way of calculating the chronometric is wrong ohlala, because this is timing on rightward choices, not timing on correct choices
        chronometric_actual = [get_median_rt(sample,a,v,isLaser,which=which,metric_type=metric_type) for a,v in itertools.product(actual_aud_azimuths,actual_vis_contrasts)]
        chronometric_actual =  np.reshape(np.array(chronometric_actual),(actual_aud_azimuths.size,actual_vis_contrasts.size)) 

        if marker is None: 
            facecolors = ['None','None','None']
        else: 
            facecolors = colors

        [ax.fill_between(vis_contrasts, l,u,color=c,alpha=.1,linestyle=line) for l,u,c in zip(c_l,c_u,colors)]
        [ax.scatter(actual_vis_contrasts,chrono,color=c,marker='o',facecolors=fc) for chrono,c,fc in zip(chronometric_actual,colors,facecolors)]

        ax.set_ylabel('%s reaction time' % metric_type)
        ax.set_xlabel('contrasts')
    
    axctrl.set_title('control')
    axopto.set_title('opto')

def av_diagnostics(sample,model=None):
    fig,ax = plt.subplots(2,3,figsize=(15,4.5),sharey=True, sharex=True)

    plt.rcParams['font.size'] = 15

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    colors = ['b','k','r']
    scaling_factor = 8
    for isLaser in range(2):
        for ia,a in enumerate(actual_aud_azimuths):
            for i,v in enumerate(actual_vis_contrasts):
                curr_cond ={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
                plot_diagnostics(model=model,sample = sample, 
                                conditions=curr_cond,data_dt =.025,method=None,myloc=i*scaling_factor,ax = ax[isLaser,ia],
                                dkwargs={'color':colors[ia],'alpha':.5},time_on_x=False)

    ax[0,0].set_ylim([.01,1.5])
    ax[0,0].set_title('ctrl trials',loc='left')
    ax[1,0].set_title('opto trials',loc='left')
    ax[0,0].set_xticks(np.arange(actual_vis_contrasts.size)*scaling_factor)
    ax[0,0].set_xticklabels(actual_vis_contrasts)
    ax[1,0].set_xlabel('contrast')
    ax[0,0].set_ylabel('reaction time (s)')
    ax[0,0].set_ylim([.1,.5])

    return fig
