# -*- coding: utf-8 -*-
"""
Example experiment. Permutation is active.
"""

from judgmentHMM import Experiment
import numpy as np

args = {"ratedTurns":['../data/complete_april_2014.csv'],
        "missing":[np.nan],
        "cleanUp":[True],
        "removePauses":[True],
        "modelParams":['SSA-ids','asr-dist','asr-conf','USA'],
        "quantizationLevel":["features"],
        "k":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "moreParams":[True],
        "validationMethod":["lolo"]
}

models = {"hmm_multi":{},
          #"hmm_laplace":{},
          #"hmm_lidstone":{"gamma":0.5},
          #"hmm_goodTuring":{},
          #"svm":{},
          "most_frequent":{}
          #"random":{}
}


exp = Experiment()
exp.run(args,models,permute=True,maxParams=4)
