# -*- coding: utf-8 -*-


import numpy as np
from numpy import nan,mean,cov,cumsum,dot,linalg,size,flipud
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten,kmeans,vq

from sklearn.metrics import *
import sklearn.cross_validation as cv
import sklearn.dummy as dummy
from sklearn.svm import SVC

import nltk as nltk
import ghmm as ghmm

import datetime
import time
import itertools
from itertools import *
import math



class Experiment:
    
    _filename = False
    _resultDir = False
    _models = False
    _wroteHeader = False
    
    def __init__(self, resultDir='results', filename = False):
        if(not filename):
            filename = datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H-%M-%S')
            filename = filename + '.csv'
                
        self._filename = filename
        self._resultDir = resultDir
        print('New experiment.')
        print('Saving results in ' + self._resultDir + '/' + self._filename)
        
    def _print(self, text):
        with open(self._resultDir + '/'+ self._filename, 'a') as f:
            f.write(text)
            
        #sys.stdout.write('.')

    # deprecated            
    def _report(self, results):
        true,pred = results.asLists()
        s = ""
        s += classification_report(true,pred)
        s += "Accuracy: " + str(accuracy_score(true,pred)) + "\n" 
        s += "Confusion matrix: " + str(confusion_matrix(true,pred))  + "\n"         
        s += "Custom: " + str(results) + "\n"
        
        return s
        
    def run(self, args, models, permute=False, maxParams=3):
        
        if(permute):
            args = self._getPermutations(args, maxParams)
            
        print('Running with ' + str(len(args)) + ' configurations')
        
        for a in args:
            a['models'] = models
            
            # if first configuration or refresh necessary (new clustering..)
            if(not self._models  or a.get('refresh')):
                print("\tpreparing data with settings: " + str(a) + ".")
                self._models = RatingModels(**a)
                
            # otherwise use old model, update set of features (=modelParams)
            else:
                print("\tsetting up new configuration.")
                if(a.get('modelParams') and a.get('quantizationLevel')
                                        and a.get('k')):
                    self._models.setModelParams(a['modelParams'],
                                                a['quantizationLevel'],
                                                a['k'])
                                                
            # run model / validate
            if(a.get('validationMethod')):
                print("\tvalidating " + str(self._models) + ".")
                self._models.validate(a['validationMethod'])

            # if new results have been produced save them
            if(a.get('validationMethod')):
                print("\tsaving results.\n")
                
                results = self._models._lastResults
                
                if(results):
                    for model,r in results.items():
                        resMap = r.getResults()
                        
                        # check whether CSV header has been added to file yet
                        if(not self._wroteHeader):
                            self._print(';'.join(map(str,
                                        resMap.keys()+a.keys()))+'\n')
                            self._wroteHeader = True
                            
                        # add a line with results from last config-run
                        self._print(';'.join(map(str,
                                        resMap.values()+a.values()))+'\n')
                        
        print("done.")

    def _getPermutations(self, args, maxParams):
        p = list()
        
        params = args["modelParams"]
        allParams = list()
        for i in range(1,maxParams+1):
            allParams += list(combinations(params,i))
        args["modelParams"] = [list(row) for row in allParams]     
        
        i=0
        for value in product(*args.values()):
            value = list(value)
            comb = dict()
            for i in range(0,len(value)):
                comb[args.keys()[i]]=value[i]
            p.append(comb)
        
        return p
        




class RatingModels:
    
    _data = False
    _modelParams = False
    _trainingMask = False
    _models = {
            'hmm':{}#,
            #'hmm_smoothed':{'smoothing':'Laplace'}
        }
    _k = 4
    _lastResults = False
    
    def __init__(self, turns=False, ratings=False, ratedTurns=False,
                 delimiter=',', ratingInterval=10,
                 missing='nan', removePauses = False, cleanUp=False,
                 modelParams=False, quantizationLevel=False, k=False,
                 validationMethod=False, moreParams = False,
                 ratingMethod = False, models=False):

        self._lastResults = False
        
        if(not ratingMethod and ratedTurns):
            self._data = DataPreparator(**{'ratedTurns':ratedTurns,
                                           'delimiter':delimiter})
        else:
            self._data = DataPreparator(**{'turns':turns,'ratings':ratings,
                                           'delimiter':delimiter})
            self._data.assignRatings(ratingMethod, ratingInterval, missing)
            
        if(models):
            self._models = models
        if(removePauses):
            self._data.removePauses()        
        if(cleanUp):
            self._data.cleanUp()
        if(moreParams):
            self._data._generateMoreParameters()
        if(modelParams):
            self.setModelParams(modelParams, quantizationLevel, k)
        if(validationMethod):
            self.validate(validationMethod)
            
    def _getCV(self, scheme):
        ''' Get cross-validation mask. For schemes see
        http://scikit-learn.org/stable/modules/cross_validation.html '''
        mask = False
        for case in switch(scheme):
            if case('lolo'):
                mask = cv.LeaveOneLabelOut(
                            self._data.getData()['label'].values)
                break
            if case('loo'):
                mask = cv.LeaveOneOut(
                            len(self._data.getData()['label'].values))
                break
                
        return mask
        
    def setModels(self, models):
        self._models = models        
        
    def setRatingMethod(self, ratingMethod):
        self._data.changeRatingAssignment(ratingMethod)
            
    def setModelParams(self, params, quantizationLevel=False, k=4):
        self._data.setModelParams(params)
        if(k):
            self._k = k
        if(quantizationLevel and quantizationLevel!="none"):
           self._data.quantize(k, quantizationLevel)
        for case in switch(quantizationLevel):
            if case('features'):
                self._modelParams = ['cluster']
                break
            if case('parameters'):
                self._modelParams = list()
                for p in params:
                    self._modelParams.append('clustered-'+p)
                break
            if case('none') or case(False):
                self._modelParams = params
                break
        
    def _train(self, model, data):

        genMod = False
        
        for case in switch(model):
            if case('hmm'):
                genMod = self._trainHMM(data)
                break
            
            if case('hmm_multi'):
                genMod = self._trainHMMMulti(data)
                break
            
            if case('svm'):
                genMod = self._trainSVM(data)
                break
            
            if case('hmm_smoothed'):
                genMod = self._trainHMM_nltk_supervised2(data)
                break
            
            if case('hmm_laplace'):
                genMod = self._trainHMM_nltk_supervised2(
                                    data,smoothing='Laplace')
                break

            if case('hmm_lidstone'):
                genMod = self._trainHMM_nltk_supervised2(
                                    data,smoothing='Lidstone')
                break                
            
            if case('hmm_goodTuring'):
                genMod = self._trainHMM_nltk_supervised2(
                                    data,smoothing='GoodTuring')
                break
            
            if case('hmm_goodTuring'):
                genMod = self._trainHMM_nltk_supervised2(
                                    data,smoothing='GoodTuring')
                break     

            if case('hmm_wittenBell'):
                genMod = self._trainHMM_nltk_supervised2(
                                    data,smoothing='WittenBell')
                break  
            
            if case('most_frequent'):
                genMod = self._trainDummy(data,'most_frequent')
                break
            
            if case('random'):
                genMod = self._trainDummy(data,'uniform')
                break
            
        return genMod
        
    def _mlHMM_multi(self, data, stateAlphabet, eAlphabet,
                     returnNltk=True):
        """
        Calculate maximum likelihood estimates for the HMM parameters
        transition probabilites, emission probabilites and initial state
        probabilites.
        Combines multiple emission features, assuming conditional independence:
        P(feat1=a & feat2=b|state) = P(feat1=a|state) * P(feat2=b|state)
        
        @param data obj of type DataPreparator that contains column 'rating'
        @param stateAlphabet list of states
        @param emissionAlphabet dict of emission alphabets for each feature
                {"feature1":[a,b,c,d], ...}
        @param returnNltk whether to return structure in nltk required format
        """
        
        features = eAlphabet.keys()
        emissions = eAlphabet.values()
        
        p = lambda feat: self._mlHMM(data,stateAlphabet,
                                     emissionAlphabet=eAlphabet[feat],
                                     emissionParameters=feat,
                                     asDict=True)
                                     
        # calculate conditional probabilites for each feature & corresp.
        # emission alphabet entry..
        # P(feat_i=emm_ij|state_k) forall: I features, J_i emissions, K states 
        # ps = {feature:emmission distribution}
        emission_ps = [p(f)[1] for f in features]
        
        p_trans,p_emmiss0,p_init = self._mlHMM(data,stateAlphabet,
                                     emissionAlphabet=eAlphabet[features[0]],
                                     emissionParameters=features[0],
                                     returnNltk=True)
        
        combined_emission_ps = {state:dict() for state in stateAlphabet}
        combined_alphabet = list()
        # create all combined probabilities by multiplying individual prob.
        # assume types nltk.CondFreqDist / nltk.FreqDist
        for emission_combination in list(itertools.product(*emissions)):
            combined_alphabet.append(tuple(emission_combination))
            for state in stateAlphabet:
                p_e = False
                for emission,dist in itertools.izip(
                                            emission_combination,emission_ps):
                    if(not p_e):
                        p_e = dist[state][emission]
                    else:
                        p_e *= dist[state][emission]
                
                combined_emission_ps[state][tuple(emission_combination)] = p_e
                
        if(returnNltk):
            return combined_alphabet,p_trans,self._dictToNLTKDistribution(
                                                combined_emission_ps),p_init
        else:
            raise NotImplementedError    
                
                
    def _dictToNLTKDistribution(self, d):
        dist = nltk.ConditionalFreqDist()
        for state in d.keys():
            for emission in d[state].keys():
                dist[state].inc(emission,
                                count=round(d[state][emission]*100000000))
        return dist
        
        
    def _mlHMM(self, data, stateAlphabet, emissionAlphabet=False,
                           emissionParameters=False, returnNltk=False,
                           asDict=False):
        """
        Calculate maximum likelihood estimates for the HMM parameters
        transitions probabilites, emission probabilites, and initial state
        probabilites.
        
        @param data obj of type DataPreparator that contains column 'rating'
        @param stateAlphabet list of states
        
        """

        if(emissionAlphabet is False):
            emissionAlphabet = xrange(self._k)
            
        if(emissionParameters is False):
            emissionParameters = self._modelParams
            
        nltk_transitions = nltk.ConditionalFreqDist()
        nltk_emissions = nltk.ConditionalFreqDist()
        nltk_prior = nltk.FreqDist()
        
        st = self._data.getSequences(data,'rating')
        #print("used train seqs: "+str(len(st)))
        
        em = self._data.getSequences(data,emissionParameters)
        
        # initialize matrices
        states_count = {a: 0 for a in stateAlphabet}
        trans_abs = {a: 0 for a in stateAlphabet}
        trans_ind = {a: {} for a in stateAlphabet}
        transitions = {a: {} for a in stateAlphabet}
        emissions = {a: {} for a in stateAlphabet}
        init = {a: 0 for a in stateAlphabet}
        for s in stateAlphabet:
            trans_ind[s] = {a: 0 for a in stateAlphabet}
            transitions[s] = {a: 0 for a in stateAlphabet}
            emissions[s] = {a: 0 for a in emissionAlphabet}
     
        # for each state sequence
        for k,seq in enumerate(st):
            emi = em[k]
            # for each state transition
            for i, state in enumerate(seq):
                # count number of state, emission pairs
                emission = emi[i]
                emissions[state][emission] += 1
                if(returnNltk):
                    nltk_emissions[state].inc(emission)
                states_count[state] += 1
                # count number of occurences for initialization            
                if(i==0):
                    init[state] += 1
                    if(returnNltk):
                        nltk_prior.inc(state)
                # count absolute transitions
                if(i<len(seq)-1):
                    # inc count of all transitions from this state
                    trans_abs[state] += 1
                    # inc count of transitions from this state to next one
                    trans_ind[state][seq[i+1]] += 1
                    # nltk
                    if(returnNltk):
                        nltk_transitions[state].inc(seq[i+1])
                    
        # divide state, emission pairs by occurence of state
        for state in stateAlphabet:
            for emission in emissionAlphabet:
                emissions[state][emission] = (float(emissions[state][emission])
                                                / float(states_count[state]))
        
        # divide relative transitions s1->s2 by absolute s1->all
        for s1 in stateAlphabet:
            for s2 in stateAlphabet:
                transitions[s1][s2] = float(
                            trans_ind[s1][s2]) / float(trans_abs[s1])
                
        # divide number of state init occ. by number of seq
        init = {state:float(init[state])/float(len(st)) for state in init}
        
        transMat = [x.values() for x in transitions.values()]
        emissMat = [x.values() for x in emissions.values()]

        if(asDict):
            return transitions, emissions, init
        elif(returnNltk):
            return nltk_transitions, nltk_emissions, nltk_prior
        else:
            return transMat,emissMat,init.values()
        
        

    def _trainHMM(self, data):
        return self._trainHMM_nltk_supervised(data)
        
    def _trainHMMMulti(self, data):
        alphabet = {f:self._data.getData()[f].unique()
                    for f in self._modelParams}
        
        multiAlph, trans, emm, prior = self._mlHMM_multi(data,[1,2,3,4,5],
                                                         alphabet)
        
        trans = nltk.ConditionalProbDist(trans,nltk.LaplaceProbDist)
        emm = nltk.ConditionalProbDist(emm,nltk.LaplaceProbDist)
        prior = nltk.LaplaceProbDist(prior)
        
        tagger = nltk.HiddenMarkovModelTagger(multiAlph,
                                              range(1,6),
                                              trans, emm, prior)
                                                                                          
        return tagger        
        
    def _trainHMM_nltk_supervised(self, data):
        trainer = nltk.HiddenMarkovModelTrainer(range(1,6),range(0,self._k))
        seqs = self._data.getLabeledSequences(data,[self._modelParams[0],
                                                    'rating'])
        #print(seqs)        
        model = trainer.train_supervised(seqs)
        return model
        
    def _trainHMM_nltk_supervised2(self,data,smoothing='Laplace'):
        trans, emm, prior = self._mlHMM(data,[1,2,3,4,5], returnNltk=True)

        debug = False        
        
        for case in switch(smoothing):
            
            if case('Laplace'):
                trans = nltk.ConditionalProbDist(trans,nltk.LaplaceProbDist)
                emm = nltk.ConditionalProbDist(emm,nltk.LaplaceProbDist)
                prior = nltk.LaplaceProbDist(prior)
                if(debug):
                    self._printCondProbDist(trans)
                    self._printCondProbDist(emm)
                break
            
            if case('Lidstone'):
                g = self._models['hmm_lidstone']['gamma']
                trans = nltk.ConditionalProbDist(trans,nltk.LidstoneProbDist,
                                                 gamma=g)
                emm = nltk.ConditionalProbDist(emm,nltk.LidstoneProbDist,
                                               gamma=g)
                prior = nltk.LidstoneProbDist(prior,g)
                if(debug):
                    self._printCondProbDist(trans)
                    self._printCondProbDist(emm)              
                break
            
            if case('GoodTuring'):
                trans = nltk.ConditionalProbDist(trans,nltk.GoodTuringProbDist)
                emm = nltk.ConditionalProbDist(emm,nltk.GoodTuringProbDist)
                prior = nltk.GoodTuringProbDist(prior)
                if(debug):
                    self._printCondProbDist(trans)
                    self._printCondProbDist(emm)            
                break
            
            if case('WittenBell'):
                trans = nltk.ConditionalProbDist(trans,nltk.WittenBellProbDist)
                emm = nltk.ConditionalProbDist(emm,nltk.WittenBellProbDist)
                prior = nltk.WittenBellProbDist(prior)
                if(debug):
                    self._printCondProbDist(trans)
                    self._printCondProbDist(emm)                
                break            

        tagger = nltk.HiddenMarkovModelTagger(range(0,self._k),
                                              range(1,6),
                                              trans, emm, prior)
                                                                                          
                                             
        return tagger

        
    def _trainHMM_nltk_unsupervised(self, data):
        trainer = nltk.HiddenMarkovModelTrainer(range(1,6),range(0,self._k))
        seqs = self._data.getSequences(data,[self._modelParams])
        model = trainer.train_unsupervised(seqs)
        return model
        
    def _printCondProbDist(self, dist):
        print(dist)
        for cond in dist:
            print(dist[cond])
            for case in range(0,6):
                print(dist[cond].prob(case))
                
    
    def _trainHMM_ghmm_supervised(self, data):
        model = False
        
        sigma = ghmm.IntegerRange(0,self._k)
 
        P_trans, P_emm, P_init = self._mlHMM(data,[1,2,3,4,5])

        
        model = ghmm.HMMFromMatrices(sigma,
                                     ghmm.DiscreteDistribution(sigma),
                                     P_trans,
                                     P_emm,
                                     P_init)
        return model        
  
        
    def _testHMM(self, model, testData, results):
        return self._testHMM_nltk(model, testData, results)
        
    def _testHMM_nltk(self, model, testData, results):
        seqs = self._data.getSequences(testData, self._modelParams)
        true_labels = self._data.getSequences(testData,self._modelParams,
                                              rated=True)
        #print("testing..")
        #print(true_labels)
        for i in range(0,len(seqs)):
            pred_labels = model.tag(seqs[i])
            true = list()
            pred = list()
            for a in true_labels[i]:
                true.append(a[1])
            for b in pred_labels:
                pred.append(b[1])
            results.compare(true,pred)
            #print(pred_labels)
        
        return results
        
        
    def _testHMM_ghmm(self, model, testData, results):
        seqs = self._data.getSequences(testData,self._modelParams)
        #print("used test seqs: "+ str(len(seqs)))
        trainSeqs = ghmm.SequenceSet(ghmm.IntegerRange(0,self._k), seqs)
        true_labels = self._data.getSequences(testData,'rating')
        pred_labels = model.viterbi(trainSeqs)
        
        return results
        
    def _trainSVM(self, data):
        model = SVC(**self._models['svm'])
        x = np.array(data[self._data._params])
        y = np.array(data['rating'])
        model.fit(x,y)
        
        return model
        
    def _testSklearn(self, model, testData, results):
        true = testData['rating'].values.tolist()
        test = np.array(testData[self._data._params])
        pred = list(model.predict(test))
        results.compare(true,pred)
        
        return results
        
    def _trainDummy(self, data, strat):
        model = dummy.DummyClassifier(strategy=strat)
        x = np.array(data[self._data._params])
        y = np.array(data['rating'])
        model.fit(x,y)
        
        return model
        
    def __str__(self):
        ms = ",".join(self._models.keys())
        fs = ",".join(self._modelParams)
        
        return "model(s) "+ms+" with features "+fs+" [k="+str(self._k)+"]"
        
    def _test(self, modelName, model, testData, results):
            
        for case in switch(modelName):
            if case('hmm','hmm_smoothed','hmm_laplace','hmm_lidstone',
                    'hmm_goodTuring','hmm_wittenBell','hmm_multi'):
                results = self._testHMM(model, testData, results)
                break
            if case('svm') or case('most_frequent') or case('random'):
                results = self._testSklearn(model, testData, results)
                break
                
        return results
        
    def validate(self, method):
        masks = self._getCV(method)
        results = dict()
        self._lastResults = False
        for m in self._models.keys():
            results[m] = TestResults(m)

        for trainM,testM in masks:

            train = self._data.getData().loc[trainM]
            test = self._data.getData().loc[testM]
            
            # validate each model for current data fold
            for m in self._models.keys():
                model = self._train(m,train)
                results[m] = self._test(m, model, test, results[m])
            
        self._lastResults = results
            
        return results


        
class TestResults:
    
    _indicators = {'MAE':'mean','MSE':'mean','hits':'hitRatio'}
    _finalInd = {'model','accuracy','scores', 'scores_f_1', 'scores_f_2', 'scores_f_3', 'scores_f_4', 'scores_f_5', 'scores_pr_1', 'scores_pr_2', 'scores_pr_3', 'scores_pr_4', 'scores_pr_5', 'scores_rec_1', 'scores_rec_2', 'scores_rec_3', 'scores_rec_4', 'scores_rec_5'}#,'classification'}
    _results = {}
    _true = list()
    _pred = list()
    _name = False
    
    def __init__(self,name, indicators = False):
        self._name = name
        self._true = list()
        self._pred = list()
        self._results = dict()
        if(indicators):
            self._indicators = indicators
        for i in self._indicators.keys():
            self._results[i] = list()
            self._results[i+'-last'] = list()
        
    def compare(self,true,pred):
        self._true.append(true)
        self._pred.append(pred)
        for ind in self._indicators.keys():
            i = getattr(self,'_'+ind)
            self._results[ind].append(i(true,pred))
        self._compare_last(true,pred)
        
    def asLists(self):
        true = list()
        pred = list()
        for t in self._true:
            true = true + t
            
        for p in self._pred:
            pred = pred + p
            
        return true, pred
            
            
    def _compare_last(self,true,pred):
        last = len(true) -1
        true = [true[last]]
        pred = [pred[last]]
        for ind in self._indicators.keys():
            i = getattr(self,'_'+ind)
            self._results[ind+'-last'].append(i(true,pred))
        
    def _MAE(self,true,pred):
        dist = 0.0
        for i in range(0,len(true)):
            dist += abs(float(true[i])-float(pred[i]))
        return dist/(float(len(true)))
        
        
    def _MSE(self, true, pred):
        dist = 0.0
        for i in range(0,len(true)):
            dist += math.pow((float(true[i])-float(pred[i])),2)
        return dist/(float(len(true)))
        
    def _hits(self,true,pred):
        hits = 0
        count = 0
        for i in range(0,len(true)):
            if(true[i]==pred[i]):
                hits += 1
            count += 1
        return hits, count
        
    def _mean(self,values):
        return np.mean(values)
        
    def _hitRatio(self,values):
        h=0
        c=0
        for hits,count in values:
            h+=hits
            c+=count
        if(c>0):
            return float(h)/float(c)
        else:
            return 0
        
    def _accuracy(self):
        true, pred = self.asLists()
        return str(accuracy_score(true,pred))

    def _scores(self):
        true, pred = self.asLists()
        return map(list,precision_recall_fscore_support(true,pred,labels=[1,2,3,4,5],average=None))

    def _scores_f_1(self):
        s = self._scores()
        return s[2][0]

    def _scores_f_2(self):
        s = self._scores()
        return s[2][1]

    def _scores_f_3(self):
        s = self._scores()
        return s[2][2]

    def _scores_f_4(self):
        s = self._scores()
        return s[2][3]

    def _scores_f_5(self):
        s = self._scores()
        return s[2][4]

    def _scores_pr_1(self):
        s = self._scores()
        return s[0][0]

    def _scores_pr_2(self):
        s = self._scores()
        return s[0][1]

    def _scores_pr_3(self):
        s = self._scores()
        return s[0][2]

    def _scores_pr_4(self):
        s = self._scores()
        return s[0][3]

    def _scores_pr_5(self):
        s = self._scores()
        return s[0][4]

    def _scores_rec_1(self):
        s = self._scores()
        return s[1][0]

    def _scores_rec_2(self):
        s = self._scores()
        return s[1][1]

    def _scores_rec_3(self):
        s = self._scores()
        return s[1][2]

    def _scores_rec_4(self):
        s = self._scores()
        return s[1][3]

    def _scores_rec_5(self):
        s = self._scores()
        return s[1][4]
        
    def _classification(self):
        true, pred = self.asLists()
        return "\""+classification_report(true,pred)+"\""
        
    def _model(self):
        return self._name
        
        
    def __str__(self):
        s ="results for "+self._name+":\n"
        for ind,aggr in self._results.items():
            aggr = self._indicators[ind.split('-')[0]]
            val = getattr(self,'_'+aggr)(self._results[ind])
            s += "\t"+ind+"\t"+str(val)+"\n"
        return s
        
    def getResults(self,getHeader=False):
        iMap = dict()
        for ind,aggr in self._results.items():
            aggr = self._indicators[ind.split('-')[0]]
            iMap[ind] = str(getattr(self,'_'+aggr)(self._results[ind]))
            
        for i in self._finalInd:
            iMap[i] = str(getattr(self,'_'+i)())
            
        return iMap
        
    def __repr__(self):
        return str(len(self._true))
        
        



class DataPreparator:
    
    _params = ['words']
    _turns = False
    _ratings = False
    _data = False
    _ratingMethods = ['latest','latestInTurn','avgLinear',
                      'closest','closestAfter','complete',
                      'closestAfterOrLatest']
    _missingRatings = dict()
    _centroids = dict()
    
    def __init__(self,turns=False,ratings=False,ratedTurns=False,
                 data=False,delimiter=','):
        
        if(data):
            self._turns = data
        else:
            if(turns and ratings):
                self._turns = pd.read_csv(turns,sep=delimiter,quotechar="\"")
                
                # DEBUG
                self._turns['asr-conf'] = self._turns['asr-conf'].fillna(0)
                self._turns['asr-dist'] = self._turns['asr-dist'].fillna(-1)
                
                self._ratings = pd.read_csv(ratings,sep=delimiter,
                                            quotechar="\"")
            elif(ratedTurns):
                self._turns = pd.read_csv(ratedTurns,sep=delimiter,
                                          quotechar="\"")
            
        self._resetData()
        
    def _label(self, data):        
        #print(data)
        self._data['label'] = (data['VP'].apply(str) + 
                                data['task'].apply(str) +
                                data['attempt'].apply(str)        )
        
    def _resetData(self):
        self._data = self._turns.copy()
        self._missingRatings = dict()
        for m in self._ratingMethods:
            self._missingRatings[m] = 0
        self._label(self._data)
        
    def _ratingMethNames(self):
        names = list()
        for m in self._ratingMethods:
            names.append('rating-' + m)
        return names
        
    def _generateMoreParameters(self):
        d = self._data
        speakerCodes = {'s':0, 'u':1, 's u':2, 'u s': 3}
        #self._data['userWords'] = d['words'] - d['systemWords']
        self._data['length'] = d['end'] - d['start']
        self._data['speakers'] = d[['speakers']].applymap(speakerCodes.get)
        
    def _cluster(self,p,k=3,iterations=100):
        # select only requested parameters for features
        obs = whiten(np.asarray(self._data[p]))
    
        # get centroids
        centroids,_ = kmeans(obs,k,iterations)
        
        # save'em
        if(isinstance(p,basestring)):
            self._centroids[p] = centroids
        else:
            self._centroids['cluster'] = centroids
        
        # assign features to clusters
        index,_ = vq(obs,centroids)
        
        return index        
    
    def loadTurns(self, turnFile, delimiter=','):
        self._turns = pd.read_csv(turnFile,sep=delimiter,quotechar="\"")
        self._resetData()
    
    def loadRatings(self, ratingsFile, delimiter=','):
        self._ratings = pd.read_csv(ratingsFile,sep=delimiter,quotechar="\"")
        self._resetData()
        
    def assignRatings(self, method, ratingInterval=False, missing=np.nan):
        #self._reset       
        meths = self._ratingMethods
        foundRatings = dict()
        for m in meths:
            foundRatings[m] = list()
        
        # find a rating for each turn (using each method from above)
        for index,row in self._turns.iterrows():
            finder = RatingFinder(self._ratings,row,ratingInterval)
            for m in meths:
                r = getattr(finder,m)()
                if(not r):
                    self._missingRatings[m] += 1
                    r = finder.getMissing(missing)
                foundRatings[m].append(r)
        
        for m in meths:
            self._data['rating-'+m] = foundRatings[m]
            
        self._data['rating'] = foundRatings[method]
        
        return self._data
        
    def changeRatingAssignment(self,method):
        self._data['rating'] = self._data['rating-'+method]
        return self._data
        
    def setRatingAssignmentMethods(self,methods):
        self._ratingMethods = methods
        self._missingRatings = dict()
        for m in self._ratingMethods:
            self._missingRatings[m] = 0
            
    def getNumMissingRatings(self):
        return self._missingRatings
        
    def setModelParams(self,params):
        self._params = params
        
    def quantize(self,k,level='features'):
        for case in switch(level):
            if case('features'):
                self._data['cluster'] = self._cluster(self._params,k)
                break
            if case('parameters'):
                for p in self._params:
                    self._data['clustered-'+p] = self._cluster(p,k)
                break
            
        return self._data
        
    def getData(self):
        return self._data
        
    def saveData(self, filename, separator=','):
        self.getData().to_csv(filename,separator)
        
    def cleanUp(self):
        before = len(self._data)
        self._data = self._data[np.isfinite(
                                self._data['rating'])].reset_index(drop=True)
        after = len(self._data)
        return before,after
        
    def removePauses(self):
        before = len(self._data)
        self._data = self._data[
                        self._data['words-sum']>0].reset_index(drop=True)
        after = len(self._data)
        return before,after
            
    def getRatings(self):
        return self.getData()[self._ratingMethNames()]
        
            
    def hist(self,param,bns):
        n,bins,patches = plt.hist(self._data[param].values,bns)
        plt.plot(bins)
        plt.xlabel(param)
        plt.show()
        
    def scatter(self,paramA,paramB):
        plt.scatter(self._data[paramA].values,self._data[paramB].values)
        plt.xlabel(paramA)
        plt.ylabel(paramB)
        plt.show()
        
    def getSequences(self, data, params, rated=False):
        data = data.groupby(['VP','task','attempt'])
        seqs = list()
        """if(not isinstance(params,basestring) and len(params)==1):
            params = params[0]
        
        if(isinstance(params, basestring)):
            for name,group in data:
                seqs.append(group[params].values.tolist())
                
        elif(len(params)>1):"""
        
        if isinstance(params,basestring):
            params = [params]
            
        for name,group in data:
            vals = list()
            for p in params:
                vals.append(group[p].values.tolist())
            
            features = map(None,*vals)
            
            if(rated):
                ratings = group['rating'].values.tolist()
                features = map(None,*[features,ratings])
            
            seqs.append(features)
                
        
        return seqs
        
    def getLabeledSequences(self, data, params, labelFirst=False):
        data = data.groupby(['VP','task','attempt'])
        seqs = list()
        for name, group in data:
            # get seq as [(a1,b1),(a2,b2)] list
            seq = list()
            for vals in group[params].values.tolist():
                if(not labelFirst):
                    t=(vals[0],vals[1])
                else:
                    t=(vals[1],vals[0])
                seq.append(t)
            seqs.append(seq)
        return seqs
        
        
        
        
        
class RatingFinder:
    
    _interval = False
    
    def __init__(self, ratings, turn, interval=10):
        self._r = ratings[['rating','time']][
                    (ratings['VP']==turn['VP']) &
                    (ratings['task'] == turn['task']) &
                    (ratings['attempt'] == turn['attempt'])]
        self._t = turn
        self._interval = interval
        
    def _fmt(self,rs):
        r = False
        if len(rs)>0:
            r = rs['rating'].values[0]
        return r
        
    def _get(self, left=False, right=False, select='all', p=['rating']):
        r = self._r[p]
        if(left):
            r = r[self._r['time'] >= (self._t[left[0]] + left[1])]
        if(right):
            r = r[self._r['time'] <= (self._t[right[0]] + right[1])]
        if(select=='first'):
            r = self._fmt(r.head(1))
        elif(select=='last'):
            r = self._fmt(r.tail(1))
        elif(isinstance(select, (int, long))):
            r = r[p].head(select)
        return r
        
    def latest(self):
        """Find the rating that is closest to the turn end."""
        
        r = self._get(left=False,right=['end',0],select='last')
        return(r)

    def latestInTurn(self):
        """Find the latest rating done after turn start."""
        
        r = self._get(left=['start',0],right=['end',0],select='last')
        return(r)
        
    def avgLinear(self):
        """Average over all ratings in turn, linear-ascendingly weighted."""
        
        r = self._get(left=['start',0],right=['end',0],p=['rating','time'])
        sumT = 0
        sumR = 0
        for i,rat in r.iterrows():
            t = rat['time'] - self._t['start']
            sumT += t
            sumR += rat['rating'] * t
            
        if(sumT>0):
            return(int(round(sumR / sumT)))
        else:
            return(False)
            
    def closest(self):
        """Find the rating closest to the end of the turn in given interval."""
        
        r = self._get(left=['end',-self._interval],
                      right=['end',self._interval],
                      p=['rating','time'])
        r['dist'] = r['time'] - self._t['end']
        r = r.sort('dist')
        return(self._fmt(r.tail(1)))
        
    def closestAfter(self):
        """Find closest rating after turn, in a given interval."""
        
        r = self._get(left=['end',0],right=['end',self._interval],
                      p=['rating','time'])
        r['dist'] = r['time'] - self._t['end']
        r = r.sort('dist')
        return(self._fmt(r.tail(1)))
        
    def complete(self):
        r = self.closestAfter()
        if not r:
            r = self.closest()
        return r
        
    def closestAfterOrLatest(self):
        r = self.closestAfter()
        if not r:
            r = self.latest()
        if not r:
            r = 3.0
        return r


    def avgInterval(self):
        """Average over all ratings in interval around the end of the turn."""
        
        r = self._get(left=['end',-self._interval],
                      right=['end',self._interval])
        r = r['rating'].values
        if len(r)>0:
            r = r.mean()
        else:
            r = False
            
    def getMissing(self,method):
        r = nan
        for case in switch(method):
            if case('1','2','3','4','5'):
                r = int(method)
            else:
                r = nan
        return r         
        
        
"""
@author: Brian Beck
"""
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

class tools:

    """ performs principal components analysis 
        (PCA) on the n-by-p data matrix A
        Rows of A correspond to observations, columns to variables. 

    Returns :  
     coeff :
       is a p-by-p matrix, each column containing coefficients 
       for one principal component.
     score : 
       the principal component scores; that is, the representation 
       of A in the principal component space. Rows of SCORE 
       correspond to observations, columns to components.
     latent : 
       a vector containing the eigenvalues 
       of the covariance matrix of A.
    """
    def princomp(A,numpc=0):
       # computing eigenvalues and eigenvectors of covariance matrix
       M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
       [latent,coeff] = linalg.eig(cov(M))
       p = size(coeff,axis=1)
       idx = argsort(latent) # sorting the eigenvalues
       idx = idx[::-1]       # in ascending order
       # sorting eigenvectors according to the sorted eigenvalues
       coeff = coeff[:,idx]
       latent = latent[idx] # sorting eigenvalues
       if numpc < p and numpc >= 0:
          coeff = coeff[:,range(numpc)] # cutting some PCs if needed
       score = dot(coeff.T,M) # projection of the data in the new space
       return coeff,score,latent
