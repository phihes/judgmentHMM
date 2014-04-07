"""
Train a hidden Markov model with Gaussian mixture models created from turn features' principal components. Returns cross-validation scores / performance metrics, see judgmentHMM.TestResults.

Command line options:
0 -- cross-validation method: loo / kfolds
1 -- number of principal components to use
2 -- number of Gaussian mixtures to use
3 -- (if kfolds) number of folds to use for CV

Example. K-fold CV with two folds, GMM with three Gaussians, data represented by four principal components:

$python gmmhmm.py kfolds 4 3 2

@author Philipp Heß
"""

from judgmentHMM import TestResults
import numpy as np
import pandas as pd
from sklearn.hmm import GMMHMM
from sklearn.mixture import GMM
import sys
import hmmParams as hmm
import sklearn.cross_validation as cv

# settings
method = sys.argv[1]
num_pc = int(sys.argv[2])
cov_type = "diag"
num_mixc = int(sys.argv[3])
p = 2
if method=="kfolds":
   flds = int(sys.argv[4])

# data
data1 = pd.read_csv('../data/complete_april_2014.csv')
mask = cv.LeaveOneLabelOut(data1['label'].values)
# mask = cv.LeavePLabelOut(data1['label'].values,p)

# select only rating + principal component data
pcas = ["pc-"+str(num_pc)+"-"+str(i) for i in xrange(1,num_pc+1)]
features = pcas
features.extend(["rating","label"])
data2 = data1[features]

# train GMM + HMM
def train(trainingData):
   # train one GMM for each state
   mixes = list()
   for state in xrange(1,6):
      # select data with current state label
      d = trainingData[trainingData.rating==state]
      # prepare data shape
      d = np.array(zip(*[d[f].values for f in pcas]))
      # init GMM
      gmm = GMM(num_mixc,cov_type)
      # train
      gmm.fit(d)
      mixes.append(gmm)

   # train HMM with init, trans, GMMs=mixes
   init,trans = hmm.hmmMlParams(trainingData,[1,2,3,4,5])
   model = GMMHMM(n_components=5,init_params='',gmms=mixes)
   model.transmat_ = trans
   model.startprob_ = init

   return model   

# test results
def test(model, testData, resultObj):
   
   # restrict test data to principal component features
   test = np.array(testData[pcas])

   # predict a dialog sequence using test data   
   # sklearn counts from 0 so add 1...
   pred = [int(r)+1 for r in list(model.predict(test))]

   # extract true ratings from test data
   true = [int(rating) for rating in testData['rating'].values.tolist()]

   #print(true)
   #print(pred)
   #print(model.score(test))

   resultObj.compare(true,pred)
   sys.stdout.flush()
       
   return resultObj

# run LOOCV
def validate(allData):
   results = TestResults("GMMHMM")

   for trainMask,testMask in mask:
      trainingData = allData.loc[trainMask]
      model = train(trainingData)
   
      testData = allData.loc[testMask]
      # leave p labels out
      for label,testGroup in testData.groupby("label"): 
         results = test(model, testGroup, results)
      sys.stdout.write('.')
   
   return results

def validateKFolds(d,folds):
   results = TestResults("GMMHMM")
   labels = list(np.unique(d['label'].values))

   for tr,te in cv.KFold(len(labels),n_folds=folds):

      trainD = d[d['label'].isin([labels[i] for i in tr])]
      testD = d[d['label'].isin([labels[i] for i in te])]

      model = train(trainD)

      for label,testGroup in testD.groupby("label"): 
         results = test(model, testGroup, results)

      sys.stdout.write('.')

   return results

# run!
if method=="loo":
   results = validate(data2)
if method=="kfolds":
   results = validateKFolds(data2,flds)
print(results.getResults())
