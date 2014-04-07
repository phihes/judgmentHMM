import pylab as pl
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import sklearn.cross_validation as cv
from sklearn.metrics import *
import sys


data = pd.read_csv("data/combined.CAOL.delta.pca.scaled.ssa-dummies.csv")

# shuffle data a bit
#data = data.reindex(np.random.permutation(data.index))

pc = int(sys.argv[1])
ssa = sys.argv[2]

if(ssa == 1):
   ssa = True
else:
   ssa = False

print("linear regression with " + str(pc) + " PCs and SSA=" + str(ssa))

features = list([False,False,False,False,False,False,False,False,False])

features[2] = ["pc-2-1","pc-2-2"]
features[3] = ["pc-3-1","pc-3-2","pc-3-3"]
features[4] = ["pc-4-1","pc-4-2","pc-4-3","pc-4-4"]
features[5] = ["pc-5-1","pc-5-2","pc-5-3","pc-5-4","pc-5-5"]
features[6] = ["pc-6-1","pc-6-2","pc-6-3","pc-6-4","pc-6-5","pc-6-6"]
features[7] = ["pc-7-1","pc-7-2","pc-7-3","pc-7-4","pc-7-5","pc-7-6","pc-7-7"]
features[8] = ["pc-8-1","pc-8-2","pc-8-3","pc-8-4","pc-8-5","pc-8-6","pc-8-7","pc-8-8"]

features = features[pc]

if(ssa):
   features.extend(["SSA-WELCOME","SSA-ASKCONFIRM","SSA-ASKFORINFO","SSA-SORRY","SSA-INFO","SSA-DETAILS","SSA-NAV"])

feat_data = [data[f].values for f in features]
rate_data = data['rating'].values



# generate CV mask
masks = cv.LeaveOneLabelOut(data['label'].values)

# store results here
true = list()
pred = list()
pred_q = list()

for train,test in masks:
   X_train = np.array(zip(*[data[f].loc[train].values for f in features]))
   X_test = np.array(zip(*[data[f].loc[test].values for f in features]))
   y_train = np.array(data['rating'].loc[train].values)
   y_test = np.array(data['rating'].loc[test].values)


   # Create linear regression object
   regr = linear_model.LinearRegression()

   # Train the model using the training sets
   regr.fit(X_train, y_train)

   true.extend(y_test)
   pred.extend(regr.predict(X_test))

   #print(y_test)
   #print(regr.predict(X_test))

# "quantize" prediction
# MSE
for i in range(0,len(true)):
   q = round(pred[i])
   if(q==0):
      q = 1
   if(q==6):
      q = 5
   pred_q.append(q)

# MAE
dist = 0.0
for i in range(0,len(true)):
   dist += abs(float(true[i])-float(pred_q[i]))
print("MAE: " + str(dist/(float(len(true)))))

# MAE-LR
dist = 0.0
for i in range(0,len(true)):
   dist += abs(float(true[i])-float(pred[i]))
print("MAE-LR: " + str(dist/(float(len(true)))))

# MSE
dist = 0.0
for i in range(0,len(true)):
   dist += (float(true[i])-float(pred[i]))**2
#print("MSE: " + str(dist/(float(len(true)))))


# R2
#print"R2: " + str((r2_score(true,pred)))

# MAE-class
for c in xrange(1,6):
   dist = 0.0
   count = 0
   for i in range(0,len(true)):
      if(true[i]==c):
         count += 1
         dist += (float(true[i])-float(pred[i]))**2
   #print("MAE-" + str(c) + ": " + str(dist/(float(count))))

#print(precision_recall_fscore_support(true,pred_q))
print(f1_score(true,pred_q, average=None))
print("acc: " + str(accuracy_score(true,pred_q)))



# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean square error
#print("Residual sum of squares: %.2f"
#      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs
#pl.scatter(X_test, diabetes_y_test,  color='black')
#pl.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
        #linewidth=3)

#pl.xticks(())
#pl.yticks(())

#pl.show()
