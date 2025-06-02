"""
script that includes model components for fitting.
# each class is a container for parameter components. Each class must contain the following objects: 

required_parameters: list of str (n)
required_conditions: str (m)
fittable_minvals: list of floats (n)
    minval for fitting parameter
fittable_maxvals: list of floats (n)
    maxval for fitting parameter
fixedC: list of floats (n)
    constant value the param should take 
freeP: list of floats (n)
    whether parameter should take the constant or the fittable 

"""

import numpy as np 
import pyddm


### some control model trials

class AV_drift(pyddm.Drift):
    name = "AV_ddm"
    required_parameters = [
        "aR", "vR","aL","vL","gamma", "b", #parameters applied to all (6)
        'coherent','conflict'
        ]
    required_conditions = ["audDiff", "visDiff"]
    fittable_minvals = [
        .01,.01,.01,.01,0.1,-4, # drift parameters
        -5,-5] 
    fittable_maxvals = [
        10,10,10,10,1.5,4, # drift parameters
        5,5]
    fixedC = [1,1,1,1,1,0,
              0,0]

    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        
        # variables
        a_R = (audSide>0)
        a_L = (audSide<0)
        v_R = (visSide>0) * (visContrast**self.gamma)
        v_L = (visSide<0) * (visContrast**self.gamma)

        audComponent = self.aR * a_R - self.aL * a_L
        visComponent = self.vR  * v_R - self.vL * v_L
        biasComponent = self.b 

        conherent  = self.coherent * (a_R * v_R) - self.coherent * (a_L * v_L)
        confict  = self.conflict * (a_R * v_L) - self.conflict * (a_L * v_R)

        myDrift = audComponent + visComponent + biasComponent + conherent + confict

        return myDrift

class visIC(pyddm.ICPoint):
    name = "starting point allowing a differential bound to a and v."
    required_parameters = ["x0", "vis_x0"]
    required_conditions = ["visDiff"]
    fittable_minvals = [-.9,-.9]
    fittable_maxvals = [.9,.9]
    fixedC = [0,0]
    freeP = [1,0]

    def get_starting_point(self, conditions):
        visSide = np.sign(conditions["visDiff"])
        start = self.x0+(self.vis_x0*visSide)
        # we fix bound and if .95 exceeded we fix the start        
        if start>0.95:
            start=.95
        elif start<-.95:
            start =-.95
        return start

class NonDecision_AudDom(pyddm.OverlayNonDecision):
    name = "Separate non-decision time for aud and vis components"
    required_parameters = ["nondectime","aud_nondec"]
    required_conditions = ["audDiff"] 
    fittable_minvals = [.01,-.4]
    fittable_maxvals = [.4,.4]
    fixedC = [.3,0]
    freeP = [1,0]    

    def get_nondecision_time(self, conditions):
        audTrial= np.abs(np.sign((conditions["audDiff"])))
        return self.nondectime  + self.aud_nondec * audTrial

############# opto models ########

class DriftAdditiveOpto(pyddm.Drift):
    name = "DriftAdditiveSplit"    
    required_parameters = [
        "aR", "vR","aL","vL","gamma", "b", #parameters applied to all (6)
        "d_aR","d_aL", "d_vR","d_vL","d_b"  #opto dependent parameters (5)
        ]    
    required_conditions = ["audDiff", "visDiff",'is_laserTrial']

    fittable_minvals = [
        .01,.01,.01,.01,0.1,-4,
        -10,-10,-10,-10,-10]
    fittable_maxvals = [
        15,15,15,15,1.5,4, 
        1,1,1,1,10]
    fixedC = [1,1,1,1,1,0,
              0,0,0,0,0]
    
    freeP = [1,1,1,1,1,0,
             0,0,0,0]

    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        isOpto = conditions['is_laserTrial']
        
        # variables
        a_R = (audSide>0)
        a_L = (audSide<0)
        v_R = (visSide>0) * (visContrast**self.gamma)
        v_L = (visSide<0) * (visContrast**self.gamma)

        audComponent = (self.aR + self.d_aR * isOpto) * a_R - (self.aL + self.d_aL * isOpto) * a_L
        visComponent = (self.vR + self.d_vR * isOpto) * v_R - (self.vL + self.d_vL * isOpto) * v_L
        biasComponent = self.b + self.d_b * isOpto

        myDrift = audComponent + visComponent + biasComponent

        return myDrift

class BoundOpto(pyddm.Bound): 
    name = 'constant bound that can expland by opto'
    required_parameters = ["B", "d_B"]
    required_conditions = ["is_laserTrial"] 
    fittable_minvals = [.9,0]
    fittable_maxvals = [1.1,4]
    fixedC = [1,0]

    def get_bound(self,conditions,*args,**kwargs):
        isOpto = conditions['is_laserTrial']

        return  self.B + (self.d_B * isOpto)

class OverlayNonDecisionOpto(pyddm.OverlayNonDecision):
    name = "Separate non-decision time for aud and vis components"
    required_parameters = ["nondectime", "d_nondectimeOpto"]
    required_conditions = ["is_laserTrial"] 
    fittable_minvals = [.01,-.4]
    fittable_maxvals = [.4,.4]
    fixedC = [.3,0]
    freeP = [1,0]    

    def get_nondecision_time(self, conditions):
        isOpto = conditions['is_laserTrial']
        return self.nondectime  + self.d_nondectimeOpto * isOpto

class ICPointOpto(pyddm.ICPoint):
    name = "A starting point with a left or right bias."
    required_parameters = ["x0",'vis_x0', "d_x0"]
    required_conditions = ["visDiff","is_laserTrial"]
    fittable_minvals = [-.9,-.9,-.9]
    fittable_maxvals = [.9,.9,.9]
    fixedC = [0,0,0]
    freeP = [1,1,0]

    def get_starting_point(self, conditions):
        isOpto = conditions['is_laserTrial']
        visSide = np.sign(conditions["visDiff"])

        start = self.x0+(self.vis_x0*visSide)+(self.d_x0*isOpto)
        # we fix bound and if .95 exceeded we fix the start        
        if start>0.95:
            start=.95
        elif start<-.95:
            start =-.95
        return start

class OverlayExponentialMixtureOpto(pyddm.Overlay):
    """An exponential mixture distribution where the mixture coef depends on opto

    """
    name = "Exponential distribution mixture model (lapse rate)"
    required_parameters = ["pmixturecoef", "rate","d_pmixturecoef"]
    required_conditions = ["is_laserTrial"]
    fittable_minvals = [0.01,.01,-.5]
    fittable_maxvals = [1,2,.5]
    fixedC = [.2,1,0]
    freeP = [1,1,0]

    def apply(self, solution):

        assert self.pmixturecoef >= 0 and self.pmixturecoef <= 1
        choice_upper = solution.choice_upper
        choice_lower = solution.choice_lower
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        # To make this work with undecided probability, we need to
        # normalize by the sum of the decided density.  That way, this
        # function will never touch the undecided pieces.
        isOpto = cond['is_laserTrial']

        norm = np.sum(choice_upper)+np.sum(choice_lower)
        lapses = lambda t : 2*self.rate*np.exp(-1*self.rate*t)
        X = m.dt * np.arange(0, len(choice_upper))
        Y = lapses(X)
        Y /= np.sum(Y)
        total_mixture = self.pmixturecoef + self.d_pmixturecoef * isOpto

        # basically we allow mixture to really saturate
        if total_mixture<=0:
            total_mixture=.0001

        choice_upper = choice_upper*(1-total_mixture) + .5*total_mixture*Y*norm # Assume numpy ndarrays, not lists
        choice_lower = choice_lower*(1-total_mixture) + .5*total_mixture*Y*norm
        #print(choice_upper)
        #print(choice_lower)
        return pyddm.Solution(choice_upper, choice_lower, m, cond, undec, evolution)
    
def get_default_noise():
    c = pyddm.NoiseConstant
    c.fittable_minvals = [.2]
    c.fittable_maxvals = [5]
    c.fixedC = [1]
    c.freeP = [1]
    return c

def get_default_drift():
    c = pyddm.DriftConstant
    c.fittable_minvals = [.01]
    c.fittable_maxvals = [3]
    c.fixedC = [.5]
    c.freeP = [1]
    return c

def get_default_bound():
    c = pyddm.BoundConstant
    c.fittable_minvals = [1]
    c.fittable_maxvals = [3]
    c.fixedC = [1]
    c.freeP = [0]
    return c

def get_default_IC():
    c = pyddm.ICPoint
    c.fittable_minvals = [-.9]
    c.fittable_maxvals = [.9]
    c.fixedC = [0]
    c.freeP = [0]
    return c

def get_default_nondecision():
    c = pyddm.OverlayNonDecision
    c.fittable_minvals = [.01]
    c.fittable_maxvals = [.4]
    c.fixedC = [1]
    c.freeP = [1]
    return c

def get_default_mixture():
    c = pyddm.OverlayExponentialMixture
    c.fittable_minvals = [.01,.1]
    c.fittable_maxvals = [.3,2]
    c.fixedC = [.01,1]
    c.freeP = [1,1]
    return c

def get_freeP_ctrl(which='ctrl'):

    if which == 'ctrl':        
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,0],
            'mixture':[0,0],
                'IC': [1,0]
        }

    elif which == 'full':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,1],
            'mixture':[1,1],
                'IC': [1,1]
        }

    elif which == 'g_coherent':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,0], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,0],
            'mixture':[1,1],
                'IC': [1,0]
        }

    elif which == 'g_conflict':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,1], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,0],
            'mixture':[1,1],
                'IC': [1,0]
        }
    
    elif which == 'g_vis_x0':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,0],
            'mixture':[1,1],
                'IC': [1,1]
        }

    elif which == 'g_aud_nondec':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,1],
            'mixture':[1,1],
                'IC': [1,0]
        }

    elif which == 'l_coherent':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,1], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,1],
            'mixture':[1,1],
                'IC': [1,1]
        }

    elif which == 'l_conflict':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,0], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,1],
            'mixture':[1,1],
                'IC': [1,1]
        }

    elif which == 'l_vis_x0':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,1],
            'mixture':[1,1],
                'IC': [1,0]
        }

    elif which == 'l_aud_nondec':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1], 
            'noise':[1],
            'bound': [0],
            'nondectime':[1,0],
            'mixture':[1,1],
                'IC': [1,1]
        }
    
    return freePs

# so, maybe we need to have anothe get_freeP for th econtrol model
def get_freeP_opto(which = 'ctrl'):
    """
    hardcoded dictionaries that allow sets of parameters to fix vs fit 

    optogenetics paramters we are going to test:
    d_aR, d_aL, d_vR, d_vL, d_b, d_nondectimeOpto, d_x0, d_pmixturecoef

    # plus some other models....


    """
    if which == 'ctrl': 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }

    elif which == 'full': 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'g_d_aR':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }

    elif which == 'g_d_aL':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,1,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }

    elif which == 'g_d_vR':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,1,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }
    
    elif which == 'g_d_vL':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,1,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }
    
    elif which == 'g_d_b':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }
    
    elif which == 'g_d_nondec':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }

    elif which == 'g_d_x0':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,1]
        }
    
    elif which == 'g_d_mixture':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,1],
                'IC': [1,1,0]
        }

    elif which == 'g_d_bound': # this is increase of the bound
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,1],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }

    elif which == 'g_d_boundx0': # this asymetric bound 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,1],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,1]
        }

    elif which == 'g_d_sensory': # this allows all sensory to be opto dependent
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,0]
        }

    elif which == 'g_d_bx0':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1,1]
        }

    elif which == 'l_d_aR':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,1,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }
    
    elif which == 'l_d_aL':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,0,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'l_d_vR':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,0,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'l_d_vL':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,0,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'l_d_b':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'l_d_nondec':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'l_d_x0':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,0]
        }

    elif which == 'l_d_mixture':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,0],
                'IC': [1,1,1]
        }

    elif which == 'l_d_sensory':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,1]
        }

    elif which == 'l_d_bx0':
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1,0]
        }

    return freePs

def get_freeP_set(fit_type='ctrl'):
    """
    get the freeP dictionary for the model to fit. 
    """
    if fit_type == 'ctrl':

        freeP_sets = [
            'ctrl',
            'full',
            'g_coherent', 
            'g_conflict',
            'g_vis_x0',
            'g_aud_nondec', 
            'l_coherent', 
            'l_conflict', 
            'l_vis_x0', 
            'l_aud_nondec', 
        ]
    
    elif fit_type == 'opto':
        freeP_sets = [
            'ctrl',
            'full',
            'g_d_aR', 
            'g_d_aL', 
            'g_d_vR', 
            'g_d_vL', 
            'g_d_b', 
            'g_d_nondec', 
            'g_d_x0', 
            'g_d_mixture',
            'g_d_bound',
            'g_d_boundx0',
            'g_d_sensory',
            'g_d_bx0',
            'l_d_aR', 
            'l_d_aL', 
            'l_d_vR', 
            'l_d_vL', 
            'l_d_b', 
            'l_d_nondec', 
            'l_d_x0', 
            'l_d_mixture',
            'l_d_sensory',
            'l_d_bx0'
        ] 

    

    return freeP_sets
