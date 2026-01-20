
import pyddm
import numpy as np


## noises
def noise_(isOpto,noise,d_noise):
    """
    noise is the standard deviation of the noise
    d_noise is the difference in noise between ctrl and opto
    """
    return noise + d_noise * isOpto

def lazy_noise(lazy_rate,t):
    return 1/np.exp(lazy_rate*t)

def lazy_noise_opto(lazy_rate,d_lazy_rate,t,isOpto):
    total_lazy_rate = lazy_rate + d_lazy_rate * isOpto
    return 1/np.exp(total_lazy_rate*t)

# other possible noise function with the lazy noise

## drifts
def vanilla_drift_(isOpto,b, d_b):

    return b + d_b * isOpto

def simple_drift_(S,isOpto,s,b,d_b):
    return s*S + b + d_b * isOpto

# currently unused
def av_drift_(a_R, v_R, a_L, v_L, isOpto, aR, vR, aL, vL, gamma, b, d_aR, d_vR, d_aL, d_vL, d_b):
    
    v_L = v_L**gamma
    v_R = v_R**gamma
    
    audComponent = (aR + d_aR * isOpto) * a_R - (aL + d_aL * isOpto) * a_L
    visComponent = (vR + d_vR * isOpto) * v_R - (vL + d_vL * isOpto) * v_L
    biasComponent = b + d_b * isOpto

    myDrift = audComponent + visComponent + biasComponent

    return myDrift

def av_drift_lazy(a_R, v_R, a_L, v_L, isOpto, aR, vR, aL, vL, gamma, b, d_aR, d_vR, d_aL, d_vL, d_b,lazy_rate,gain,t):
    
    v_L = v_L**gamma
    v_R = v_R**gamma
    
    audComponent = (aR + d_aR * isOpto) * a_R - (aL + d_aL * isOpto) * a_L
    visComponent = (vR + d_vR * isOpto) * v_R - (vL + d_vL * isOpto) * v_L
    biasComponent = b + d_b * isOpto

    myDrift = audComponent + visComponent + biasComponent


    return myDrift * lazy_noise(gain,lazy_rate,t)

# used to compare the original AV model with pulse model
def av_drift_pulse(a_R, v_R, a_L, v_L, isOpto, aR, vR, aL, vL, gamma, b, d_b, a_pulse_end,t):
    v_L = v_L**gamma
    v_R = v_R**gamma
    
    audComponent = (aR) * a_R - (aL) * a_L
    visComponent = (vR) * v_R - (vL) * v_L
    biasComponent = b + d_b * isOpto

    myDrift = ((audComponent if t < a_pulse_end else 0) + 
               (visComponent)+ 
               biasComponent)


    return myDrift

def av_drift_lazy_pulse(a_R, v_R, a_L, v_L, isOpto, aR, vR, aL, vL, gamma, b, d_b,lazy_rate,t, a_pulse_end):
    v_L = v_L**gamma
    v_R = v_R**gamma
    
    audComponent = (aR) * a_R - (aL) * a_L
    visComponent = (vR) * v_R - (vL) * v_L
    biasComponent = b + d_b * isOpto

    myDrift = ((audComponent if t < a_pulse_end else 0) + 
               (visComponent)+ 
               biasComponent)


    return myDrift * lazy_noise(lazy_rate,t)  

def av_drift_optoLazy(a_R, v_R, a_L, v_L, isOpto, aR, vR, aL, vL, gamma, bias, d_bias,lazy_rate,d_lazy_rate,t, a_pulse_end):
    v_L = v_L**gamma
    v_R = v_R**gamma
    
    audComponent = (aR) * a_R - (aL) * a_L
    visComponent = (vR) * v_R - (vL) * v_L
    biasComponent = bias + d_bias * isOpto

    myDrift = ((audComponent if t < a_pulse_end else 0) + 
               (visComponent)+ 
               biasComponent)


    return myDrift * lazy_noise_opto(lazy_rate,d_lazy_rate,t,isOpto)

# unilaterally modifyable starting point and bound
def x0_(isOpto, B,d_BL, x0, d_x0):
    
    BR = B
    BL = B

    BL_ = BL + d_BL * isOpto

    zero_start = (2*BL_)/(BL_+BR)-1     
    
    start = zero_start + x0 + d_x0 * isOpto
    # we fix bound and if .95 exceeded we fix the start        
    if start > 0.95:
        start = .95
    elif start < -.95:
        start = -.95
    return start

def Bound_(B,d_BL,isOpto): 
    
    # bound
    BR = B
    BL = B
    
    BL_ = BL + d_BL * isOpto
    bound = (BR + BL_)/2
    return bound

# bilaterally modifyable bound and starting point
def x0_bilateral(isOpto, B,d_Bound, x0, d_x0):

    B_ = B + d_Bound * isOpto

    BR = B_
    BL = B_


    zero_start = (2*BL)/(BL+BR)-1
    
    start = zero_start + x0 + d_x0 * isOpto
    # we fix bound and if .95 exceeded we fix the start        
    if start > 0.95:
        start = .95
    elif start < -.95:
        start = -.95
    return start

def Bound_bilateral(B,d_Bound,isOpto):
    # bound
    B_ = B + d_Bound * isOpto

    BR = B_
    BL = B_
    
    bound = (BR + BL)/2
    return bound

def ndtime_(isOpto, nondec, d_nondec):
    # nondec is the time before the decision process starts
    # d_nondec is the difference in nondec time between ctrl and opto
    ndtime = nondec + d_nondec * isOpto
    return ndtime

def mixture_(isOpto,umixture,d_umixture):
    return umixture + d_umixture * isOpto

## these two are reeally just helper functions for handling parameters
def split_to_delta_params(parameters):

    changing_params = [param for param in parameters.keys() if param.startswith('d_')]
    ctrl_params = [param for param in parameters.keys() if not param.startswith('d_')]


    ctrl_params = {param: parameters[param] for param in ctrl_params}
    varying_params = {param: parameters[param] for param in changing_params}
    fixed_params = {param:0 for param in changing_params}

    return ctrl_params, varying_params, fixed_params

def get_delta_param_sets(parameters,which = 'opto'):
    """helper function that produces paramet sets to fit
    the logic is as flollows: 
    in get predefined_param_bounds, you need to choose which parameters you allow to vary or which are always fitted the same way 

    here we combine them into gain and loss models to assess their effects on the model fit


    """
    
    is_opto = 'opto' in which
    is_bilateral = 'bilateral' in which

    ctrls,bounds,fixes = split_to_delta_params(parameters)

    # to get model
    ctrl = {'ctrl':{**ctrls,**fixes}}
    
    if is_opto:
        full = {'full': {**ctrls,**bounds}}

        varying_params = list(bounds)
        varying_params = {param:[param] for param in varying_params}

        
        if is_bilateral:
            param_combos = {
                #'d_sensory': ['d_aR','d_vR','d_aL','d_vL'],
                'd_biasBound': ['d_bias','d_Bound'],
                'd_x0Bound': ['d_x0','d_Bound'],
                'd_Boundlazy': ['d_Bound','d_lazy_rate'],
            }

        else:
        
            param_combos = {
                #'d_sensory': ['d_aR','d_vR','d_aL','d_vL'],
                'd_bx0BL': ['d_b','d_x0','d_BL'],
                'd_bx0': ['d_x0','d_b'],
                'd_bBL': ['d_b','d_BL'],
                'd_x0BL': ['d_x0','d_BL'],
            }

        reduced_models ={**varying_params,**param_combos}

        gain_models = {}
        loss_models = {}
        for model,params_of_interest in reduced_models.items():
            gain_models[f'g_{model}'] = {**ctrls,
                                                **{k: v for k, v in bounds.items() if k in params_of_interest},
                                                **{k: v for k, v in fixes.items() if k not in params_of_interest}}
            
            loss_models[f'l_{model}'] = {**ctrls,
                                                **{k: v for k, v in bounds.items() if k not in params_of_interest},
                                                **{k: v for k, v in fixes.items() if k in params_of_interest}}

        models_to_fit = {**ctrl,**full,**gain_models,**loss_models}
    
    
    else:
        models_to_fit = {**ctrl}


    return models_to_fit

# potentially can get rid of T_dur as a parameter ... 
def my_hyperparams():
    hyperparams = {
        'conditions':['v_L','v_R','a_L','a_R','isOpto'],
        'dt':.01,
        'dx': .01,
        'T_dur' : 1.5, 
        'choice_names':('Right','Left')
        }
    
    return hyperparams


def get_model(which='av_original'):

    if which == 'av_original':
        modelfunctions = {
            'drift': av_drift_pulse,
            'noise': noise_,
            'nondecision': ndtime_,
            'starting_position': x0_,
            'bound': Bound_,
            'mixture_coef': 'umixture'
        }

        full_model_params = {
                'aR': (0,10),
                'vR': (0,10),
                'aL': (0,10),
                'vL': (0,10),
                'a_pulse_end':(0.01,0.5),
                "gamma":(0.3,1.5),
                'b': (-5,5),
                'x0': (-.7,.7),
                'nondec': (0,.6),
                'noise': 1,
                'B':((.1,3)),
                'umixture': (0,1),
                'd_noise':(-1,1),
                'd_b': (-10,10),
                'd_x0': (-.9,.9),
                'd_nondec': (-.3,.3),
                "d_BL":(-.5,5),
            }  
        
    if (which == 'av_lazy')|(which == 'av_opto_unilateral'):
        modelfunctions = {
            'drift': av_drift_lazy_pulse,
            'noise': lazy_noise,
            'nondecision': ndtime_,
            'starting_position': x0_,
            'bound': Bound_,
            'mixture_coef': 'umixture'
        }

        full_model_params = {
                'aR': (0,10),
                'vR': (0,10),
                'aL': (0,10),
                'vL': (0,10),
                'a_pulse_end':(0.01,0.5),
                "gamma":(0.3,1.5),
                'b': (-5,5),
                'x0': (-.7,.7),
                'nondec': (0,.6),
                'lazy_rate':(1,10),
                'umixture': (0,1),
                'B':((.1,3)),
                'd_b': (-5,5),
                'd_x0': (-.4,.4),
                'd_nondec': (-.3,.3),
                "d_BL":(0,3),
            }

    if which == 'av_lazy_opto_bilateral': 
        modelfunctions = {
            'drift': av_drift_optoLazy,
            'noise': lazy_noise_opto,
            'nondecision': ndtime_,
            'starting_position': x0_bilateral,
            'bound': Bound_bilateral,
            'mixture_coef': 'umixture'
        }

        full_model_params = {
                'aR': (0,10),
                'vR': (0,10),
                'aL': (0,10),
                'vL': (0,10),
                'a_pulse_end':(0.01,0.5),
                "gamma":(0.3,1.5),
                'bias': (-5,5),
                'x0': (-.7,.7),
                'nondec': (0,.6),
                'lazy_rate':(1,10),
                'umixture': (0,1),
                'B':((.1,3)),
                'd_bias': (-5,5),
                'd_x0': (-.4,.4),
                'd_nondec': (-.3,.3),
                "d_Bound":(0,3),
                'd_lazy_rate':(0,5),
            }
        
    if which== 'av_opto_bilateral': 
        modelfunctions = {
            'drift': av_drift_pulse,
            'noise': lazy_noise,
            'nondecision': ndtime_,
            'starting_position': x0_bilateral,
            'bound': Bound_bilateral,
            'mixture_coef': 'umixture'
        }

        full_model_params = {
                'aR': (0,10),
                'vR': (0,10),
                'aL': (0,10),
                'vL': (0,10),
                'a_pulse_end':(0.01,0.5),
                "gamma":(0.3,1.5),
                'bias': (-5,5),
                'x0': (-.7,.7),
                'nondec': (0,.6),
                'umixture': (0,1),
                'B':((.1,3)),
                'd_bias': (-10,10),
                'd_x0': (-.9,.9),
                'd_nondec': (-.3,.3),
                "d_Bound":(-.5,5),
            }


    hyperparams = my_hyperparams()

    return modelfunctions, full_model_params, hyperparams



