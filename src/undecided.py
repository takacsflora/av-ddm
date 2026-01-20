# specific functions for the likelihood

import numpy as np
import pyddm
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from paranoid.types import Self, Number, Positive0, Natural1
from pyddm.sample import Sample
from pyddm.model import Model
from pyddm.logger import logger as _logger

    

class LossLikelihoodUndecided(pyddm.LossFunction):
    """Likelihood loss function"""
    name = "Negative log likelihood"
    _robustness_param = 0
    @staticmethod
    def _test(v):
        assert v.dt in Positive0()
        assert v.T_dur in Positive0()
    @staticmethod
    def _generate():
        yield LossLikelihoodUndecided(sample=next(Sample._generate()), dt=.01, T_dur=3)
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        # Each element in the dict is indexed by the conditions of the
        # model (e.g. coherence, trial conditions) as a frozenset.
        # Each contains a tuple of lists, which are to contain the
        # position for each within a histogram.  For instance, if a
        # reaction time corresponds to position i, then we can index a
        # list representing a normalized histogram/"pdf" (given by dt
        # and T_dur) for immediate access to the probability of
        # obtaining that value.
        self.hist_indexes = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            s = self.sample.subset(**comb)
            maxt = max(max(s.choice_upper) if s.choice_upper.size != 0 else -1, max(s.choice_lower) if s.choice_lower.size != 0 else -1)
            assert maxt <= self.T_dur, "Simulation time T_dur=%f not long enough for these data. (max sample RT=%f)" % (self.T_dur, maxt)
            # Find the integers which correspond to the timepoints in
            # the pdfs.  Also don't group them into the first bin
            # because this creates bias.
            choice_upper = [int(round(e/dt)) for e in s.choice_upper]
            choice_lower = [int(round(e/dt)) for e in s.choice_lower]
            undec = s.undecided
            self.hist_indexes[frozenset(comb.items())] = (choice_upper, choice_lower, undec)
    @accepts(Self, Model)
    @returns(Number)
    @requires("model.dt == self.dt and model.T_dur == self.T_dur")
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            # nans come from negative values in the pdfs, which in
            # turn come from the dx parameter being set too low.  This
            # comes up when fitting, because sometimes the algorithm
            # will "explore" and look at extreme parameter values.
            # For example, this arises when standard deviation is very
            # close to 0.  We will issue a warning now, but throwing
            # an exception may be the better way to handle this to
            # make sure it doesn't go unnoticed.
            with np.errstate(all='raise', under='ignore'):
                try:
                    loglikelihood += np.sum(np.log(sols[k].pdf("_top")[self.hist_indexes[k][0]] + self._robustness_param))
                    loglikelihood += np.sum(np.log(sols[k].pdf("_bottom")[self.hist_indexes[k][1]] + self._robustness_param))
                except FloatingPointError:
                    minlike = min(np.min(sols[k].pdf("_top")), np.min(sols[k].pdf("_bottom")))
                    if minlike == 0:
                        _logger.warning("Infinite likelihood encountered. Please either use a Robust likelihood method (e.g. LossRobustLikelihood or LossRobustBIC) or even better use a mixture model (via an Overlay) which covers the full range of simulated times to avoid infinite negative log likelihood.  See the FAQs in the documentation for more information.")
                    elif minlike < 0:
                        _logger.warning("Infinite likelihood encountered. Simulated histogram is less than zero in likelihood calculation.  Try decreasing dt.")
                    _logger.debug(model.parameters())
                    return np.inf
            # # This is not a valid way to incorporate undecided trials into a likelihood
            if sols[k].prob_undecided() > 0:
               loglikelihood += np.log(sols[k].prob_undecided()/model.dt)*self.hist_indexes[k][2]
               
        return -loglikelihood
    


