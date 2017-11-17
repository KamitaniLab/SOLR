function [ model ] = updateBeta(X,y,model)
%UPDATEBETA Summary of this function goes here
%   Detailed explanation goes here

    effectiveDim=find(model.alpha<10.^8);
    initialBeta=model.beta(effectiveDim);
    func4fminunc=@(beta)func2minimize(X,y,model,beta,effectiveDim);
    %options = optimoptions('fminunc','GradObj','on','Algorithm','trust-region');
    options = optimset('Gradobj','on','Hessian','on','Display','off');
    [bestBeta4effectiveDim,funcVal,dummy,dummy,dummy,hessValue ] = fminunc(func4fminunc,initialBeta,options);
    %[bestBeta4effectiveDim] = fminsearch(func4fminunc,initialBeta,options);
    
    newBeta=zeros(size(X,2),1);
    newCovBeta=zeros(size(X,2),size(X,2));
    cov4effectiveDim=inv(hessValue);
    for index_effectiveDim=1:length(effectiveDim)
        newBeta(effectiveDim(index_effectiveDim))=bestBeta4effectiveDim(index_effectiveDim);
        newCovBeta(effectiveDim(index_effectiveDim),effectiveDim)=...
            cov4effectiveDim(index_effectiveDim,:);
    end
    model.beta=newBeta;
    model.covBeta=newCovBeta;
end



function [funcValue gradValue hessValue] = func2minimize(X,y,model,betaAsArgument,effectiveDim)
    y_1ofK=zeros(size(y,1),1+length(model.mu));
    for index_class=1:size(y_1ofK,2)
        y_1ofK(y==index_class,index_class)=1;
    end

    alpha=model.alpha(effectiveDim);
    %alpha=diag(alpha);
    mu=model.mu;

    beta=zeros(size(X,2),1);
    for index_effectiveDim=1:length(effectiveDim)
        beta(effectiveDim(index_effectiveDim))=betaAsArgument(index_effectiveDim);
    end


    [P F f df]=calcPredictiveProbability(X(:,effectiveDim),betaAsArgument,mu);
    P(P<10.^(-150))=10.^(-150);
    funcValue=y_1ofK.*log(P);
    funcValue=sum(funcValue(:));
    funcValue=funcValue-0.5.*((beta')*alpha*beta);%add ARD prior effect
    funcValue=-funcValue;
    
    gradValue=zeros(length(effectiveDim),1);
    for index_sample=1:size(y,1)
        if y(index_sample)==1
            dlogP_dbeta=f(index_sample,y(index_sample));
        else
            dlogP_dbeta=f(index_sample,y(index_sample))-f(index_sample,y(index_sample)-1);
        end
        dlogP_dbeta=dlogP_dbeta./P(index_sample,y(index_sample));
        dlogP_dbeta=dlogP_dbeta.*(-X(index_sample,effectiveDim)');
        gradValue=gradValue+dlogP_dbeta;
    end
    %add ARD prior
    ARDterm=alpha*beta;
    ARDterm=ARDterm(effectiveDim,:);
    gradValue=gradValue-ARDterm;
    gradValue=-gradValue;
                              
                              
                              
    hessValue=zeros(length(effectiveDim),length(effectiveDim));
    for index_sample=1:size(y,1)
        if y(index_sample)==1
            d2logP_dbetabeta=df(index_sample,y(index_sample))./P(index_sample,y(index_sample)) - ...
                (f(index_sample,y(index_sample))./P(index_sample,y(index_sample))).^2;
        else
            d2logP_dbetabeta=(df(index_sample,y(index_sample))-df(index_sample,y(index_sample)-1))./P(index_sample,y(index_sample)) - ...
                ((f(index_sample,y(index_sample))-f(index_sample,y(index_sample)-1))./P(index_sample,y(index_sample))).^2;
        end
        d2logP_dbetabeta=d2logP_dbetabeta.*((X(index_sample,effectiveDim)')*X(index_sample,effectiveDim));
        hessValue=hessValue+d2logP_dbetabeta;
    end
    %add ARD prior
    ARDterm=alpha;
    ARDterm=ARDterm(effectiveDim,:);
    ARDterm=ARDterm(:,effectiveDim);
    hessValue=hessValue-ARDterm;
    hessValue=-hessValue;
end
