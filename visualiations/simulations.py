#%%

import pyddm
import numpy as np 
import matplotlib.pyplot as plt



plt.rcParams['font.size'] = 12

class DriftOnly(pyddm.Drift):
    name = "drift"
    required_parameters = ["vis","bias"]
    required_conditions = ["C"]
    def get_drift(self, conditions, **kwargs):
        return (self.vis * conditions["C"]) + self.bias

def get_rt_quartiles(m,v,correct_only = True,left_only = False,right_only = False):
    sol = m.solve(conditions={"C":v})

    if correct_only:
        correct_side = np.sign(v)
    elif left_only:
        correct_side = -1
    elif right_only:
        correct_side = 1
    else:
        correct_side = 0 # when we want to get both

    percentiles = [.25,.5,.75]
    l = [np.interp(np.ptp(sol.cdf('Left'))*p,sol.cdf('Left'),sol.t_domain) for p in percentiles]
    r = [np.interp(np.ptp(sol.cdf('Right'))*p,sol.cdf('Right'),sol.t_domain) for p in percentiles]
    j = [np.interp(np.ptp(sol.cdf('Right')+sol.cdf('Left'))*p,sol.cdf('Right')+sol.cdf('Left'),sol.t_domain) for p in percentiles]

    if correct_side==-1: 
        lower,mid,upper = l
    elif correct_side==1: 
        lower,mid,upper = r
    elif correct_side==0: 
        lower,mid,upper = j

    return mid

def plot_sym_psychometric(bias=0,vis_coef=8,noise=2,bound=1,x0=0,n_simulated=2):

    biases, vis_coefs,noises,bounds,x0s = (np.linspace(*param, n_simulated) if isinstance(param, (list, tuple)) else np.ones(n_simulated)*param 
                         for param in (bias, vis_coef,noise,bound,x0))



    fig,(axp,ax1,ax2,axc,axc1) = plt.subplots(1,5,figsize=(15,3))
    vis_contrasts = np.linspace(-1,1,40)
    colors = plt.cm.inferno(np.linspace(0.2,.8,x0s.size))
    
    for i,(bias,vis_coef,noise,bound,x0) in enumerate(zip(biases, vis_coefs,noises,bounds,x0s)):
        m = pyddm.Model(drift=DriftOnly(vis=vis_coef,bias=bias),
                        noise=pyddm.NoiseConstant(noise=noise),
                        bound=pyddm.BoundConstant(B=bound),
                        overlay=pyddm.OverlayChain(overlays=[
                            pyddm.OverlayNonDecision(nondectime=.3),
                            pyddm.OverlayExponentialMixture(pmixturecoef=0,
                            rate=1),
                            ]),
                        IC=pyddm.ICPoint(x0=x0),
                        dt=.01, dx=.01, T_dur=1.5,choice_names = ('Right','Left'))


        pdf_r = m.solve(conditions={"C":0}).pdf('Right')
        axp.plot(m.t_domain(),(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])
        #axp.plot(m.t_domain(),(-1)*m.solve(conditions={"C":0}).cdf('Left'),color=colors[i])


        psychometric = np.array([m.solve(conditions={"C": v}).prob('Right') for v in vis_contrasts])
        psychometric_log = np.log10(psychometric/(1-psychometric))

        ax1.plot(vis_contrasts,psychometric_log,color=colors[i])

        ax2.plot(vis_contrasts,psychometric,color=colors[i])
        # chronometric
        #chronometric = ([m.solve(conditions={"C":v}).mean_decision_time() for v in vis_contrasts])
        # chronometric_c = [get_rt_quartiles(m,v,correct_only=False) for v in vis_contrasts]
        # chronometric_e = [get_rt_quartiles(m,v,correct_only=True) for v in vis_contrasts]

        chronometric_c = [get_rt_quartiles(m,v,correct_only=False,left_only=True) for v in vis_contrasts]
        chronometric_e = [get_rt_quartiles(m,v,correct_only=False, right_only=True) for v in vis_contrasts]

        axc.plot(vis_contrasts,chronometric_c,color=colors[i])
        axc1.plot(vis_contrasts,chronometric_e,color=colors[i])


    ax1.axhline(0,color='k',linestyle='--')
    ax1.axvline(0,color='k',linestyle='--')

    axp.set_xlabel('time (s)')
    axp.set_ylabel('pdf at 0 contrast')

    ax1.set_xlabel('contrast')
    ax1.set_ylabel('log odds')

    ax2.set_xlabel('contrast')
    ax2.set_ylabel('p(R)')

    axc.set_xlabel('contrast')
    axc.set_ylabel('median reaction time on correct')

    axc1.set_xlabel('contrast')
    axc1.set_ylabel('median reaction time on incorrect') 

    plt.show()

def plot_pdfs(bias=0,vis_coef=8,noise=2,bound=1,x0=0,n_simulated=2):

    biases, vis_coefs,noises,bounds,x0s = (np.linspace(*param, n_simulated) if isinstance(param, (list, tuple)) else np.ones(n_simulated)*param 
                         for param in (bias, vis_coef,noise,bound,x0))
    # we want to simulate unilatearl bound change so we need ot correct x0 in this case 
    if isinstance(bound, (list, tuple)):
        x0s = (bounds-1)
        print(x0s)
        print(bounds)



    fig,axs= plt.subplots(1,3,figsize=(12,3),sharey=True)
    stim_strengths = [-.25,0,.25]
    colors = plt.cm.inferno(np.linspace(0.2,.8,biases.size))
    for i_s,stim_strength in enumerate(stim_strengths):
        axu = axs[i_s]
        for i,(bias,vis_coef,noise,bound,x0) in enumerate(zip(biases, vis_coefs,noises,bounds,x0s)):
            m = pyddm.Model(drift=DriftOnly(vis=vis_coef,bias=bias),
                            noise=pyddm.NoiseConstant(noise=noise),
                            bound=pyddm.BoundConstant(B=bound),
                            overlay=pyddm.OverlayChain(overlays=[
                                pyddm.OverlayNonDecision(nondectime=.3),
                                pyddm.OverlayExponentialMixture(pmixturecoef=0,
                                rate=1),
                                ]),
                            IC=pyddm.ICPoint(x0=x0),
                            dt=.01, dx=.01, T_dur=1,choice_names = ('Right','Left'))


            pdf_r = m.solve(conditions={"C":stim_strength}).pdf('Right')
            #axu.plot(m.t_domain(),(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])
            axu.plot(m.t_domain(),pdf_r,color=colors[i])

            pdf_l = m.solve(conditions={"C":stim_strength}).pdf('Left')
            
            
            axu.plot(m.t_domain(),-pdf_l,color=colors[i])

            #axu.plot(m.t_domain(),-(pdf_l-np.min(pdf_l))/(np.max(pdf_l) - np.min(pdf_l)),color=colors[i])

            sol = m.solve(conditions={"C":stim_strength})
            median_r = np.interp(np.ptp(sol.cdf('Right'))*.5,sol.cdf('Right'),sol.t_domain)
            #axu[0].axvline(median_r, color=colors[i], linestyle='--')

            median_l = np.interp(np.ptp(sol.cdf('Left'))*.5,sol.cdf('Left'),sol.t_domain)
            #axu[0].axvline(median_l, color=colors[i], linestyle='--')

            axu.vlines(median_r,1.2,1.5,color=colors[i], linestyle='-')
            axu.vlines(median_l,-1.5,-1.2,color=colors[i], linestyle='-')
            axu.set_xlim([0.25,1])


plot_sym_psychometric(bias=(0,3),vis_coef=8,noise=2,bound=1,x0=0,n_simulated=2)



plot_pdfs(bias=0,vis_coef=8,noise=2,bound=1,x0=(0,0.2),n_simulated=2)

plot_pdfs(bias=(0,3),vis_coef=8,noise=2,bound=1,x0=0,n_simulated=2)


plot_pdfs(bias=0,vis_coef=8,noise=2,bound=(1,1.2),x0=0,n_simulated=2)

# %%
plot_pdfs(bias=0,vis_coef=8,noise=(2,3),bound=1,x0=0,n_simulated=2)

# %%
plot_sym_psychometric(bias=0,vis_coef=8,noise=(2,3),bound=1,x0=0,n_simulated=2)

# %%
