
# coding: utf-8

# In[2]:

import numpy
import scipy.stats
import scipy.optimize


# In[2]:

def calcCumulativeOdds(feature,beta,mu):
    F=numpy.zeros([numpy.shape(feature)[0],len(mu)+1])#cumulative odds
    f=numpy.zeros([numpy.shape(feature)[0],len(mu)+1])#corresponding pdf
    df=numpy.zeros([numpy.shape(feature)[0],len(mu)+1])#derivative
    activationValue=numpy.dot(feature,beta)
    for index_class in range(len(mu)):
        F[:,index_class]=scipy.stats.logistic.cdf(mu[index_class]-activationValue)
    F[:,len(mu)]=1
    f=F*(1-F)
    df=F*(1-F)*(1-2*F)
    output=dict()
    output['F']=F
    output['f']=f
    output['df']=df
    return output

def calcProbability(feature,beta,mu):
    cumulativeOdds=calcCumulativeOdds(feature,beta,mu)
    F=cumulativeOdds['F']
    probability=numpy.zeros([numpy.shape(feature)[0],len(mu)+1])
    probability[:,0]=F[:,0]
    if len(mu) > 1:
        for index_class in range(1,len(mu)+1):
            probability[:,index_class]=F[:,index_class]-F[:,index_class-1]
    return probability


# In[3]:

def updateBeta(feature,label,model):
    mu=model['mu']
    alpha=model['alpha']
    effectiveDim=list()
    for index_alpha in range(len(alpha)):
        if alpha[index_alpha] < 10**8:
            effectiveDim.append(index_alpha)
    effectiveDim=numpy.array(effectiveDim)
    label_1ofK=numpy.zeros([numpy.shape(feature)[0],len(mu)+1])
    for index_sample in range(numpy.shape(feature)[0]):
        label_1ofK[index_sample,label[index_sample]]=1
        
    def func2minimize(betaAsArgument):
        beta=numpy.zeros(numpy.shape(feature)[1]);
        for index_effectiveDim in range(len(effectiveDim)):
            beta[effectiveDim[index_effectiveDim]]=betaAsArgument[index_effectiveDim]
        P=calcProbability(feature,beta,mu)
        P[P<10**(-8)]=10**(-8)
        funcValue=label_1ofK*numpy.log(P)
        funcValue=numpy.sum(numpy.sum(funcValue,axis=1),axis=0)
        #add ARD prior effect
        for index_beta in range(len(beta)):
            funcValue=funcValue-(0.5*beta[index_beta]**2)*alpha[index_beta]
        funcValue=-funcValue
        return funcValue
    
    def grad2minimize(betaAsArgument):
        beta=numpy.zeros(numpy.shape(feature)[1])
        for index_effectiveDim in range(len(effectiveDim)):
            beta[effectiveDim[index_effectiveDim]]=betaAsArgument[index_effectiveDim]
        P=calcProbability(feature,beta,mu)
        P[P<10**(-8)]=10**(-8)
        cumOdds=calcCumulativeOdds(feature,beta,mu)
        f=cumOdds['f']
        gradValue=numpy.zeros(len(effectiveDim))
        for index_sample in range(numpy.shape(feature)[0]):
            if label[index_sample]==0:
                dlogP_dbeta=f[index_sample,label[index_sample]];
            else:
                dlogP_dbeta=f[index_sample,label[index_sample]]-f[index_sample,label[index_sample]-1];
    
            dlogP_dbeta=dlogP_dbeta/P[index_sample,label[index_sample]];
            dlogP_dbeta=dlogP_dbeta*(-feature[index_sample,effectiveDim]);
        gradValue=gradValue+dlogP_dbeta;
        #add ARD term
        ARDterm=numpy.dot(numpy.diag(alpha),beta);
        ARDterm=ARDterm[effectiveDim];
        gradValue=gradValue-ARDterm;
        gradValue=-gradValue;
        return gradValue
    def Hess2minimize(betaAsArgument):
        beta=numpy.zeros(numpy.shape(feature)[1])
        for index_effectiveDim in range(len(effectiveDim)):
            beta[effectiveDim[index_effectiveDim]]=betaAsArgument[index_effectiveDim]
        P=calcProbability(feature,beta,mu)
        P[P<10**(-8)]=10**(-8)
        cumOdds=calcCumulativeOdds(feature,beta,mu)
        f=cumOdds['f']
        df=cumOdds['df']
        hessValue=numpy.zeros([len(effectiveDim),len(effectiveDim)])
        for index_sample in range(numpy.shape(feature)[0]):
            if label[index_sample]==0:
                d2logP_dbetabeta=                df[index_sample,label[index_sample]]/P[index_sample,label[index_sample]] -                     (f[index_sample,label[index_sample]]/P[index_sample,label[index_sample]])**2;
            else:
                d2logP_dbetabeta=(df[index_sample,label[index_sample]]-df[index_sample,label[index_sample]-1])/P[index_sample,label[index_sample]] -                 ((f[index_sample,label[index_sample]]-f[index_sample,label[index_sample]-1])/P[index_sample,label[index_sample]])**2
            x=feature[index_sample,effectiveDim]
            d2logP_dbetabeta=d2logP_dbetabeta*numpy.array(numpy.transpose(numpy.mat(x))*numpy.mat(x))
            hessValue=hessValue+d2logP_dbetabeta;
        #add ARD prior effect
        ARDterm=numpy.diag(alpha);
        ARDterm=ARDterm[effectiveDim,:];
        ARDterm=ARDterm[:,effectiveDim];
        hessValue=hessValue-ARDterm;
        hessValue=-hessValue;
        return hessValue
    
    initialValue=model['beta']
    initialValue=initialValue[effectiveDim]
    res = scipy.optimize.minimize(func2minimize, initialValue, method='Newton-CG',jac=grad2minimize,hess=Hess2minimize,tol=10**(-3))
    newBeta=numpy.zeros(numpy.shape(feature)[1]);
    newBeta[effectiveDim]=res['x']
    covForEffectiveDim=numpy.linalg.inv(Hess2minimize(res['x']))
    newCovBeta=numpy.zeros([numpy.shape(feature)[1],numpy.shape(feature)[1]]);
    for index_effectiveDim in range(len(effectiveDim)):
        newCovBeta[effectiveDim[index_effectiveDim],effectiveDim]=covForEffectiveDim[index_effectiveDim,:]
    model['beta']=newBeta
    model['covBeta']=newCovBeta
    return model


# In[4]:

def updateMu(feature,label,model):
    alpha=model['alpha']
    beta=model['beta']
    label_1ofK=numpy.zeros([numpy.shape(feature)[0],len(model['mu'])+1])
    for index_sample in range(numpy.shape(feature)[0]):
        label_1ofK[index_sample,label[index_sample]]=1
        
    def func2minimize(muAsArgument):
        mu=muAsArgument
        P=calcProbability(feature,beta,mu)
        P[P<10**(-8)]=10**(-8)
        funcValue=label_1ofK*numpy.log(P)
        funcValue=numpy.sum(numpy.sum(funcValue,axis=1),axis=0)
        #add ARD prior effect
        for index_beta in range(len(beta)):
            funcValue=funcValue-(0.5*beta[index_beta]**2)*alpha[index_beta]
        funcValue=-funcValue
        return funcValue
    
    def grad2minimize(muAsArgument):
        mu=muAsArgument
        P=calcProbability(feature,beta,mu)
        P[P<10**(-8)]=10**(-8)
        cumOdds=calcCumulativeOdds(feature,beta,mu)
        f=cumOdds['f']
        gradValue=numpy.zeros(len(mu))
        for index_sample in range(numpy.shape(feature)[0]):
            dlogP_dmu=numpy.zeros(len(mu))
            if label[index_sample]<len(mu):
                dlogP_dmu[label[index_sample]]=                    f[index_sample,label[index_sample]]/P[index_sample,label[index_sample]]

            if label[index_sample]!=0:
                dlogP_dmu[label[index_sample]-1]=                    -f[index_sample,label[index_sample]-1]/P[index_sample,label[index_sample]]
            gradValue=gradValue+dlogP_dmu
        gradValue=-gradValue
        return gradValue
    
    initialValue=model['mu']
    constraints=list()
    for index_constraint in range(1,len(model['mu'])):
        temporal_constraint=dict()
        temporal_constraint['type']='ineq'
        temporal_constraint['fun']= lambda x: numpy.array([x[index_constraint] - x[index_constraint-1]])
        constraints.append(temporal_constraint)
    res = scipy.optimize.minimize(func2minimize, initialValue, method='SLSQP',jac=grad2minimize,tol=10**(-3),constraints=constraints)
    model['mu']=res['x']
    return model


# In[5]:

def updateAlpha(feature,label,model):
    alpha=model['alpha']
    covBeta=model['covBeta']
    beta=model['beta']
    for index_alpha in range(len(alpha)):
        if alpha[index_alpha] < 10**8:
            #Yamashita et al. update rule
            alpha[index_alpha]=(1-alpha[index_alpha]*covBeta[index_alpha,index_alpha])/(beta[index_alpha]**2)
            #original update rule
            #alpha[index_alpha]=1/(beta[index_alpha]+covBeta[index_alpha,index_alpha])
        else:
            beta[index_alpha]=0
    model['beta']=beta
    model['alpha']=alpha
    return model


# In[ ]:



