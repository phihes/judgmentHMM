import numpy as np
import pylab as pl
import pandas as pd
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale

from mpl_toolkits.mplot3d import Axes3D

# PCA
def compute_scores(X):
   pca = PCA()
   pca_scores = []
   for n in n_components:
      pca.n_components = n
      pca.fit(X)
      pca_scores.append(pca.explained_variance_ratio_)
        #pca_scores.append(np.mean(cross_val_score(pca, X)))

   return pca_scores

# prepare data
features = ["asr-conf","asr-dist","attempt","barge-in","end","length","turn","words-system","words-user"]
data = pd.read_csv("data/combined.CAOL.delta.csv")
data = data[data["words-sum"]>0]
d = np.array(zip(*[data[f].values for f in features]))
d = scale(d)

n_components = np.arange(1,len(features)+1,1)

#pca_scores = compute_scores(d)
#n_components_pca = n_components[np.argmax(pca_scores)]

#for score in pca_scores:
#   print(np.sum(score))

pca = PCA()
pca_scores = []
for n in n_components:
   pca.n_components = n
   dn = pca.fit_transform(d)
   pca_scores.append(pca.explained_variance_ratio_)
   print(pca.components_)
   for i in range(0,n):
      data['pc-'+str(n)+'-'+str(i+1)] = dn[:,i]

print(pca_scores)
#data.to_csv('data/combined.CAOL.delta.pca.scaled.csv')
#print(d)
#print(d3)
#fig = pl.figure()
#ax = Axes3D(fig)
#ax.scatter(d3[:,0],d3[:,1],d3[:,2])
#pl.show()
