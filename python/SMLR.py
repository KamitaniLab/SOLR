"""
SMLR (sparse multinomial logistic regression)
"""






import numpy
import SMLRupdate
from sklearn.base import BaseEstimator, ClassifierMixin
class SMLR(BaseEstimator, ClassifierMixin):
    def __init__(self):
        print "SMLR (sparse multinomial logistic regression)"
    
    def fit(self,feature,label):
        #feature: matrix, whose size is # of samples by # of dimensions.
        #label: label vector, whose size is # of samples.
        #       If you treat a classification problem with C classes, please use 0,1,2,...,(C-1) to indicate classes
        
        
        #Check # of features, # of dimensions, and # of classes
        self.classes_, indices = numpy.unique(label,return_inverse=True)
        N=feature.shape[0]
        D=feature.shape[1]
        #C=numpy.max(label)+1
        #C=C.astype(int)
        C=len(self.classes_)
        
        #transoform label into a 1-d array to avoid possible errors
        label=indices
        
        
        #make class label based on 1-of-K representation
        label_1ofK=numpy.zeros((N,C))
        for n in range(N):
            label_1ofK[n,label[n]]=1
    
    
        #add a bias term to feature
        feature=numpy.hstack((feature,numpy.ones((N,1))))
        D=D+1
    
        
        #set initial values of theta (wieghts) and alpha (relavence parameters)
        theta=numpy.zeros((D,C))
        alpha=numpy.ones((D,C))
        isEffective=numpy.ones((D,C))
        effectiveFeature=range(D)
        
        #Variational baysian method (see Yamashita et al., 2008)
        for iteration in range(100):
            
            #theta-step
            newThetaParam=SMLRupdate.thetaStep(theta,alpha,label_1ofK,feature,isEffective)
            theta=newThetaParam['mu']#the posterior mean of theta
            
            #alpha-step
            alpha=SMLRupdate.alphaStep(alpha,newThetaParam['mu'],newThetaParam['var'],isEffective)
            
            #pruning of irrelevant dimensions (that have large alpha values)
            isEffective=numpy.ones(theta.shape)
            isEffective[alpha>10**3]=0
            theta[alpha>10**3]=0
            
            dim_excluded=(numpy.all(isEffective==0,axis=1))
            dim_excluded=[d for d in range(len(dim_excluded)) if dim_excluded[d]]
            theta=numpy.delete(theta,dim_excluded,axis=0)
            alpha=numpy.delete(alpha,dim_excluded,axis=0)
            feature=numpy.delete(feature,dim_excluded,axis=1)
            isEffective=numpy.delete(isEffective,dim_excluded,axis=0)
            effectiveFeature=numpy.delete(effectiveFeature,dim_excluded,axis=0)
            
            #show progress
            if iteration%20==0:
                num_effectiveWeights=numpy.sum(isEffective)
                print "# of iterations: %d ,  # of effective dimensions: %d" %(iteration,len(effectiveFeature))
    
        temporal_theta=numpy.zeros((D,C))
        temporal_theta[effectiveFeature,:]=theta
        theta=temporal_theta
                    
        self.coef_=numpy.transpose(theta[:-1,:])
        self.intercept_=theta[-1,:]
        return theta

    def predict(self,feature):
        N=feature.shape[0]
        D=feature.shape[1]
        
        #add a bias term to feature
        feature=numpy.hstack((feature,numpy.ones((N,1))))
        
        #load weights
        w=numpy.vstack((numpy.transpose(self.coef_),self.intercept_))
        C=w.shape[1]
        
        #predictive probability calculation
        p=numpy.zeros((N,C))
        predicted_label=list([])
        for n in range(N):
            p[n,:]=numpy.exp(numpy.dot(feature[n,:],w))
            p[n,:]=p[n,:]/sum(p[n,:])
            predicted_label.append(self.classes_[numpy.argmax(p[n,:])])
        return predicted_label

    def predict_proba(self,feature):
        N=feature.shape[0]
        D=feature.shape[1]
        
        #add a bias term to feature
        feature=numpy.hstack((feature,numpy.ones((N,1))))
        
        #load weights
        w=numpy.vstack((numpy.transpose(self.coef_),self.intercept_))
        C=w.shape[1]
        
        #predictive probability calculation
        p=numpy.zeros((N,C))
        predicted_label=numpy.zeros(N)
        for n in range(N):
            p[n,:]=numpy.exp(numpy.dot(feature[n,:],w))
            p[n,:]=p[n,:]/sum(p[n,:])
        return p

