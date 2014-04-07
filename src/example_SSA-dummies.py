# -*- coding: utf-8 -*-
"""
Example experiment. Use exactly one set of parameters, no permutation. Quantize features into 8 clusters. Use only HMM.
"""

from judgmentHMM import Experiment

arg = {"ratedTurns":'../data/complete_april_2014.csv',
       "validationMethod":"lolo",
       "quantizationLevel":"parameters",
       "k":8,
       "modelParams",['asr-conf','words-user',"SSA-WELCOME","SSA-ASKCONFIRM","SSA-ASKFORINFO","SSA-SORRY","SSA-INFO","SSA-DETAILS","SSA-NAV"]}

models = {"hmm_multi":{}}

exp = Experiment()
exp.run([a],models,permute=False)
