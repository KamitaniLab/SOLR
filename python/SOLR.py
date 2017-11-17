
# coding: utf-8

# In[25]:

import numpy
import SOLRsubfunc
from sklearn.base import BaseEstimator, ClassifierMixin
class SOLR(BaseEstimator, ClassifierMixin):
    def __init__(self):
        print "SOLR (sparse ordinal logistic regression)"
    
    def fit(self,feature,label):
        #feature: matrix, whose size is # of samples by # of dimensions.
        #label: label vector, whose size is # of samples.
        #       If you treat a classification problem with C classes, please use 0,1,2,...,(C-1) to indicate classes
        
        model=dict()
        model['alpha']=numpy.ones(numpy.shape(feature)[1])
        model['beta']=numpy.ones(numpy.shape(feature)[1])
        model['mu']=numpy.linspace(-1,1,numpy.max(label))
        #Variational baysian method (see Yamashita et al., 2008)
        for iteration in range(100):
            
            #beta & mu update
            for subIteration in range(3):
                model=SOLRsubfunc.updateBeta(feature,label,model)
                model=SOLRsubfunc.updateMu(feature,label,model)

            #alpha update
            model=SOLRsubfunc.updateAlpha(feature,label,model)
            
            
            #show progress
            if iteration%20==0:
                print "# of iterations: %d ,  # of effective dimensions: %d" %(iteration,len(numpy.nonzero(model['beta'])[0]))
    
        
                    
        self.coef_= model['beta']
        self.mu_= model['mu']
        return model['beta']

    def predict_proba(self,feature):
        p=SOLRsubfunc.calcProbability(feature,self.coef_,self.mu_)
        return p
    
    def predict(self,feature):
        p=SOLRsubfunc.calcProbability(feature,self.coef_,self.mu_)
        predicted_label=list([])
        for index_sample in range(numpy.shape(p)[0]):
            predicted_label.append(numpy.argmax(p[index_sample,:]))
        return predicted_label

