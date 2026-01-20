

import numpy as np
import matplotlib.pyplot as plt
import itertools
from pyddm.functions import solve_partial_conditions


def plot_diagnostics(model=None,sample = None, conditions=None,data_dt =.025,method=None,myloc=0,ax = None,
                     tlim = [0.1,0.7],
                     dkwargs = None,mkwargs =None,time_on_x = True,data_plot_func='fill_between'):
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
        
        data_top_norm = np.asarray(data_hist_top)/total_samples/data_dt
        data_bot_norm = -np.asarray(data_hist_bot)/total_samples/data_dt
        datfunc = getattr(ax, data_plot_func)



        
        if time_on_x:
            datfunc(t_domain_data,data_top_norm, label="Data", **dkwargs)
            datfunc(t_domain_data,data_bot_norm, label="Data", **dkwargs)

            undecided = s.prob_undecided()/data_dt

            maxt = min(tlim[1],T_dur)
            nogo_t_loc = maxt + 0.1
            ax.set_xlim([tlim[0],nogo_t_loc + 0.05])


            ax.plot([nogo_t_loc,nogo_t_loc],[undecided,-undecided],'o-',**dkwargs)


            #datfunc(t_domain_data,np.cumsum(data_top_norm), label="Data", **dkwargs)
            #datfunc(t_domain_data+myloc,np.cumsum(data_bot_norm), label="Data", **dkwargs) 
            ## adding undecided trials
            # ax.bar(T_dur+data_dt,s.prob_undecided,**dkwargs)
            # ax.bar(T_dur+data_dt+myloc,-s.prob_undecided,**dkwargs)
        else:
            datfunc(data_top_norm+myloc,t_domain_data, label="Data",**dkwargs)
            datfunc(data_bot_norm+myloc,t_domain_data, label="Data", **dkwargs)
            
            # adding undecided trials
            #ax.arrow(0+myloc,T_dur+data_dt,s.prob_undecided()/data_dt,0,width = data_dt*2,**dkwargs)
            #ax.arrow(0+myloc,T_dur+data_dt,-s.prob_undecided()/data_dt,0,width = data_dt*2,**dkwargs)
            undecided_x1 = myloc + s.prob_undecided()/data_dt
            undecided_x2 = myloc - s.prob_undecided()/data_dt
            undecided_y = T_dur + data_dt -np.abs(conditions['visDiff']/20)
            ax.plot([undecided_x1,undecided_x2],[undecided_y,undecided_y],'o-',markersize=1,**dkwargs)


        
        toplabel,bottomlabel = sample.choice_names
    if model:
        s = solve_partial_conditions(model, sample, conditions=conditions, method=method)

        if time_on_x:
            ax.plot(model.t_domain(),s.pdf("_top"),**mkwargs)
            ax.plot(model.t_domain(),-s.pdf("_bottom"),**mkwargs)

            # we normalise this to the data scaling (as the sum of the pdf  is the same as p_undec)
            ax.plot([nogo_t_loc,nogo_t_loc],[s.prob_undecided()/data_dt,-s.prob_undecided()/data_dt],'o',**mkwargs)

        else:
            ax.plot(s.pdf("_top")+myloc,model.t_domain(),**mkwargs)
            ax.plot(-s.pdf("_bottom")+myloc,model.t_domain(),**mkwargs)
            #ax.plot(myloc+(s.prob_undecided()/model.dt),model.t_domain()[-1],'o',markersize=10,**mkwargs)
        
        toplabel,bottomlabel = model.choice_names

def get_rt_quartiles(m,a,v,o,which = 'correct'):
    sol = m.solve(conditions={'a_R':np.abs(a)*(a>0),'a_L':np.abs(a)*(a<0),'v_R':np.abs(v)*(v>0),'v_L':np.abs(v)*(v<0),'isOpto':o})

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

def plot_psychometric(model,sample,axctrl=None,axopto=None,plot_log=False,plot_opto=True,datkws={},predkws={}): 

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    aud_azimuths  = np.linspace(-1,1,3)
    vis_contrasts = np.linspace(-1,1,40)

    if plot_opto:
        linestyles= ['--','-']
        markerstyles = [None,'filled']
        if axctrl is None or axopto is None:         
            _,(axctrl,axopto) = plt.subplots(2,1,figsize=(20,10))
        axes = [axctrl,axopto]

    else:
        linestyles= ['-']
        markerstyles = ['filled']
        if axctrl is None:        
            _,axctrl = plt.subplots(1,1,figsize=(10,5))
        axes = [axctrl]


    for isLaser,(line,marker,ax) in enumerate(zip(linestyles,markerstyles,axes)):
        psychometric,a,v = zip(*[[model.solve(conditions={'a_R':np.abs(a)*(a>0),'a_L':np.abs(a)*(a<0),'v_R':np.abs(v)*(v>0),'v_L':np.abs(v)*(v<0),'isOpto':isLaser}).prob('Right'),a,v] 
                                 for a,v in itertools.product(aud_azimuths,vis_contrasts)])
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
            [ax.plot(np.abs(vis_contrasts)**gamma * np.sign(vis_contrasts),p,color=c,linestyle=line,**predkws) for p,c in zip(psychometric_log,colors)]
            [ax.scatter(np.abs(actual_vis_contrasts)**gamma * np.sign(actual_vis_contrasts),p,color=c,facecolors=fc,**datkws) for p,c,fc in zip(psychometric_actual_log,colors,facecolors)]
            ax.set_ylim([-3,3])
            ax.axhline(0,color='k',linestyle='--')

        else:
            [ax.plot(vis_contrasts,p,color=c,linestyle=line,**predkws) for p,c in zip(psychometric,colors)]
            [ax.scatter(actual_vis_contrasts,p,color=c,facecolors=fc,**datkws) for p,c,fc in zip(psychometric_actual,colors,facecolors)]
            ax.set_ylim([-.05,1.05])
            ax.axhline(0.5,color='k',linestyle=':')

        ax.axvline(0,color='k',linestyle=':')
        ax.set_ylabel('p(R)')
        ax.set_xlabel('contrasts')
    

def plot_chronometric(model,sample,which='correct',axctrl=None,axopto=None,metric_type='median',plot_opto=True,datkws={},predkws={}):

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    aud_azimuths  = np.linspace(-1,1,3)
    vis_contrasts = np.linspace(-1,1,40)
    colors = ['b','k','r'] # for -1,0,1 aud

    if plot_opto:
        linestyles= ['--','-']
        markerstyles = [None,'filled']

        if axctrl is None or axopto is None:         
            _,(axctrl,axopto) = plt.subplots(2,1,figsize=(20,10))
        axes = [axctrl,axopto]
    
    else:
        linestyles= ['-']
        markerstyles = ['filled']
        if axctrl is None:        
            _,axctrl = plt.subplots(1,1,figsize=(10,5))
        axes = [axctrl]


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

        [ax.fill_between(vis_contrasts, l,u,color=c,alpha=.1,linestyle=line,**predkws) for l,u,c in zip(c_l,c_u,colors)]
        [ax.scatter(actual_vis_contrasts,chrono,color=c,facecolors=fc,**datkws) for chrono,c,fc in zip(chronometric_actual,colors,facecolors)]

        ax.set_ylabel('%s reaction time' % metric_type)
        ax.set_xlabel('contrasts')


def av_diagnostics(sample,model=None):
    fig,ax = plt.subplots(2,3,figsize=(15,4.5),sharey=True, sharex=True)

    plt.rcParams['font.size'] = 15

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    colors = ['b','k','r']
    scaling_factor = 12
    for isLaser in range(2):
        for ia,a in enumerate(actual_aud_azimuths):
            for i,v in enumerate(actual_vis_contrasts):
                curr_cond ={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
                plot_diagnostics(model=model,sample = sample, 
                                conditions=curr_cond,data_dt =.025,method=None,myloc=i*scaling_factor,ax = ax[isLaser,ia],
                                dkwargs={'color':colors[ia],'alpha':.5},time_on_x=False)

    ax[0,0].set_ylim([0,.5])
    ax[0,0].set_title('ctrl trials',loc='left')
    ax[1,0].set_title('opto trials',loc='left')
    ax[0,0].set_xticks(np.arange(actual_vis_contrasts.size)*scaling_factor)
    ax[0,0].set_xticklabels(actual_vis_contrasts)
    ax[1,0].set_xlabel('contrast')
    ax[0,0].set_ylabel('reaction time (s)')

    T_dur = max(sample)
    ax[0,0].set_ylim([.1,T_dur+.1])

    return fig


def ctrl_vs_opto(sample):
    
    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))[::-1]
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    colors = ['r','k','b']

    n_rows = len(actual_aud_azimuths)
    n_cols = len(actual_vis_contrasts)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5*n_cols, 3*n_rows), sharex=True, sharey=True)

    fig.subplots_adjust(wspace=-.5, hspace=-.6)
    # Set transparent background for all axes and the figure
    for ax_row in axes if n_rows > 1 else [axes]:
        for ax in ax_row if n_cols > 1 else [ax_row]:
            ax.set_facecolor('none')
    fig.patch.set_alpha(1.0)


    for i, (aud,color) in enumerate(zip(actual_aud_azimuths,colors)):
        for j, vis in enumerate(actual_vis_contrasts):
            ax = axes[i, j] if n_rows > 1 and n_cols > 1 else (axes[j] if n_rows == 1 else axes[i])
            
            
            plot_diagnostics(
                sample=sample,
                conditions={'audDiff': aud, 'visDiff': vis,'isOpto': 0},
                ax=ax,
                time_on_x=True,
                data_plot_func='plot',dkwargs={'linewidth':2,'alpha':1,'color':color}
            )

            plot_diagnostics(
                sample=sample,
                conditions={'audDiff': aud, 'visDiff': vis,'isOpto': 1},
                ax=ax,
                time_on_x=True,
                dkwargs={'color':color,'alpha':0.5,'linestyle':'--', 'linewidth':1}
            )
            


    # Remove all spines and ticks except for the left column (preserve only y-axis)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j] if n_rows > 1 and n_cols > 1 else (axes[j] if n_rows == 1 else axes[i])
            if j == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks([])
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.yaxis.set_ticks([])
                ax.xaxis.set_ticks([])

            
def plot_undecided(sample,model=None,axctrl=None,axopto=None,plot_opto=True,datkws={},predkws={}):

    visDiff_arrays = sample.conditions['visDiff']

    visDiff_combined = np.concatenate(visDiff_arrays)

    audDiff_arrays = sample.conditions['audDiff']
    audDiff_combined = np.concatenate(audDiff_arrays)


    visDiffs  = np.sort(np.unique(visDiff_combined))
    audDiffs = np.sort(np.unique(np.sign(audDiff_combined)))

    colors = ['blue','grey','red']

    if plot_opto:
        linestyles= ['--','-']
        if axctrl is None or axopto is None:         
            _,(axctrl,axopto) = plt.subplots(2,1,figsize=(2,4),sharey=True,sharex=True)
        axes = [axctrl,axopto]

    else:
        linestyles= ['-']
        if axctrl is None:        
            _,axctrl = plt.subplots(1,1,figsize=(2,2),sharey=True,sharex=True)
        axes = [axctrl]

    
    for i in range(len(axes)):
        ax = axes[i]
        for audDiff,color in zip(audDiffs,colors):
            
            data,pred = [],[]
            for visDiff in visDiffs:
                v_L = np.abs(visDiff)*(visDiff<0)
                v_R = np.abs(visDiff)*(visDiff>0)
                a_L = np.abs(audDiff)*(audDiff<0)
                a_R = np.abs(audDiff)*(audDiff>0)
                conditions = {'a_L':a_L,'v_L':v_L,'isOpto':i,'a_R':a_R,'v_R':v_R}
                sam = sample.subset(**conditions)
                data.append(sam.prob_undecided())

                if model is not None:
                    s = solve_partial_conditions(model, sample, conditions=conditions, method=None)
                    pred.append(s.prob_undecided())


            ax.plot(visDiffs,data,label='data',color=color,**datkws)
            if model is not None:
                ax.plot(visDiffs,pred,'-',label='model',color=color,**predkws)
            ax.set_xlabel('v_R')
            ax.set_ylabel('P(undecided)') 


def plot_undecided_ctrl_vs_opto(sample,ax=None):

    visDiff_arrays = sample.conditions['visDiff']

    visDiff_combined = np.concatenate(visDiff_arrays)

    audDiff_arrays = sample.conditions['audDiff']
    audDiff_combined = np.concatenate(audDiff_arrays)


    visDiffs  = np.sort(np.unique(visDiff_combined))
    audDiffs = np.sort(np.unique(np.sign(audDiff_combined)))

    colors = ['blue','grey','red']
    styles = [':', '-']
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(2,2),sharey=True,sharex=True)
    for i in range(2):
        for audDiff,color in zip(audDiffs,colors):
            
            data,pred = [],[]
            for visDiff in visDiffs:
                v_L = np.abs(visDiff)*(visDiff<0)
                v_R = np.abs(visDiff)*(visDiff>0)
                a_L = np.abs(audDiff)*(audDiff<0)
                a_R = np.abs(audDiff)*(audDiff>0)
                conditions = {'a_L':a_L,'v_L':v_L,'isOpto':i,'a_R':a_R,'v_R':v_R}
                sam = sample.subset(**conditions)
                data.append(sam.prob_undecided())

            ax.plot(visDiffs,data,label='data',color=color,linestyle=styles[i])
            ax.set_xlabel('v_R')
            ax.set_ylabel('P(undecided)') 



def plot_av_diagnostics2(sample, model=None, plot_ctrl=True, plot_opto=False,xlims=[.03, .8],
                         ctrl_kws={'color':'k','alpha':.7},
                         opto_kws={'color':'orange','alpha':.7},
                         mkwargs = {'color':'k','lw':1}):
    """_summary_

    Args:
        samm (_type_): _description_
    """

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))

    n_rows = len(actual_aud_azimuths)
    n_cols = len(actual_vis_contrasts)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(.5*n_cols, .75*n_rows),dpi=150, sharex=True, sharey=True)

    fig.subplots_adjust(wspace=0.02, hspace=-.05)
    # Set transparent background for all axes and the figure
    for ax_row in axes if n_rows > 1 else [axes]:
        for ax in ax_row if n_cols > 1 else [ax_row]:
            ax.set_facecolor('none')
    fig.patch.set_alpha(1.0)


    for ia,a in enumerate(actual_aud_azimuths):
            for i,v in enumerate(actual_vis_contrasts):

                current_ax = axes[n_rows-ia-1, i] if n_rows > 1 else axes[i]

                common_kws = {
                    'data_dt':.03,
                    'sample':sample,
                    'ax':current_ax,
                    'time_on_x':True,
                    'tlim':xlims

                }

                if model is None:
                

                    if plot_ctrl:
                        conditions = {'audDiff': a, 'visDiff': v, 'isOpto': 0}
                        
                        plot_diagnostics(model=None,conditions=conditions,
                            data_plot_func='plot',dkwargs=ctrl_kws,**common_kws)

                    if plot_opto:
                        conditions = {'audDiff': a, 'visDiff': v, 'isOpto': 1}
                    
                        plot_diagnostics(model=None,conditions=conditions,
                        data_plot_func='plot',dkwargs=opto_kws,**common_kws)
                        
                    current_ax.axhline(0, color='black', lw=.5, linestyle=':')
                        
                    

                if model is not None:
                    if plot_ctrl:
                        conditions = {'audDiff': a, 'visDiff': v, 'isOpto': 0}
                        dkwargs = ctrl_kws
                    elif plot_opto:
                        conditions = {'audDiff': a, 'visDiff': v, 'isOpto': 1}
                        dkwargs = opto_kws

                    plot_diagnostics(model=model,conditions=conditions,
                        data_plot_func='fill_between',dkwargs=dkwargs,mkwargs=mkwargs,**common_kws)
                    
                #current_ax.set_xlim(xlims)


    

    # Remove all spines and ticks except for the left column (preserve only y-axis)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j] if n_rows > 1 and n_cols > 1 else (axes[j] if n_rows == 1 else axes[i])
            if j == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks([])
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.yaxis.set_ticks([])
                ax.xaxis.set_ticks([])

    return fig 
