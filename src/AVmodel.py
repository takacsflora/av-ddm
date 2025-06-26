
import pyddm

def simple_drift_(S,isOpto,s,b,d_b):

    return s*S + b + d_b * isOpto

def drift_(a_R, v_R, a_L, v_L, isOpto, aR, vR, aL, vL, gamma, b, d_aR, d_vR, d_aL, d_vL, d_b):
    
    v_L = v_L**gamma
    v_R = v_R**gamma
    
    audComponent = (aR + d_aR * isOpto) * a_R - (aL + d_aL * isOpto) * a_L
    visComponent = (vR + d_vR * isOpto) * v_R - (vL + d_vL * isOpto) * v_L
    biasComponent = b + d_b * isOpto

    myDrift = audComponent + visComponent + biasComponent

    return myDrift

def x0_(isOpto, BL,BR,d_BL, x0, d_x0):
    

    BL_ = BL + d_BL * isOpto

    zero_start = (2*BL_)/(BL_+BR)-1
    
    start = zero_start + x0 + d_x0 * isOpto
    # we fix bound and if .95 exceeded we fix the start        
    if start > 0.95:
        start = .95
    elif start < -.95:
        start = -.95
    return start

def ndtime_(isOpto, nondec, d_nondec):
    # nondec is the time before the decision process starts
    # d_nondec is the difference in nondec time between ctrl and opto
    ndtime = nondec + d_nondec * isOpto
    return ndtime

def Bound_(BL,BR,d_BL,isOpto): 
    
    BL_ = BL + d_BL * isOpto
    bound = (BR + BL_)/2
    return bound

def mixture_(isOpto,umixture,d_umixture):
    return umixture + d_umixture * isOpto

def all_params():
      return {
        'aR': (0,30),
        'vR': (0,30),
        'aL': (0,30),
        'vL': (0,30),
        "gamma":(0.1,2.5),
        'b': (-10,10),
        'x0': (-.9,.9),
        'nondec': (0,.6),
        'noise': (1.5,5),
        'umixture': (0,1),
        'BL':1,
        'BR':1,
        'd_aR': (-10,1),
        'd_vR': (-10,1),
        'd_aL': (-10,1),
        'd_vL': (-10,1),
        'd_b': (-10,10),
        'd_x0': (-.9,.9),
        'd_nondec': (-.3,.3),
        "d_BL":(-.5,5),
    }  

def my_hyperparams():
    hyperparams = {
        'conditions':['v_L','v_R','a_L','a_R','isOpto'],
        'dt':.001,
        'dx': .001,
        'T_dur' : 0.6, 
        'choice_names':('Right','Left')
        }
    
    return hyperparams

def split_to_delta_params(parameters):

    changing_params = [param for param in parameters.keys() if param.startswith('d_')]
    ctrl_params = [param for param in parameters.keys() if not param.startswith('d_')]


    ctrl_params = {param: parameters[param] for param in ctrl_params}
    varying_params = {param: parameters[param] for param in changing_params}
    fixed_params = {param:0 for param in changing_params}

    return ctrl_params, varying_params, fixed_params

def get_param_sets():
    """helper function that produces paramet sets to fit
    the logic is as flollows: 
    in get predefined_param_bounds, you need to choose which parameters you allow to vary or which are always fitted the same way 

    here we combine them into gain and loss models to assess their effects on the model fit


    """

    parameters = all_params()
    ctrls,bounds,fixes = split_to_delta_params(parameters)

    # to get model
    ctrl = {'ctrl':{**ctrls,**fixes}}
    full = {'full': {**ctrls,**bounds}}

    varying_params = list(bounds)
    varying_params = {param:[param] for param in varying_params}

    param_combos = {
        'd_sensory': ['d_aR','d_vR','d_aL','d_vL'],
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


    return models_to_fit

def assemble_model(params,drift_type = 'av'):
    
    hyperparams = my_hyperparams()

    if drift_type == 'av':
        driftfun = drift_
    elif drift_type == 'simulated':
        driftfun = simple_drift_
    
    m = pyddm.gddm(drift=driftfun,
                noise="noise",
                starting_position=x0_,
                bound=Bound_,
                nondecision= ndtime_,
                mixture_coef = "umixture",
                parameters=params,**hyperparams)
    
    return m



