# -*- coding: utf-8 -*-
"""
Example experiment. Train HMMs with PC 1 to 8, k=3..20.
"""

from judgmentHMM import Experiment

arg = {"ratedTurns":'../data/complete_april_2014.csv',
       "validationMethod":"lolo",
       "quantizationLevel":"parameters"}
models = {"hmm_multi":{}}
args = list()

for k in xrange(3,21):
    for total in xrange(2,9):
        newarg = arg.copy()
 
        params = list()
        for cur in xrange(1,total+1):
            params.append("pc-"+str(total)+"-"+str(cur))
        newarg["modelParams"] = params
        newarg["k"] = k
        args.append(newarg)
    
exp = Experiment()
exp.run(args,models,permute=False)
