function [ model ] = updateBeta_v4(X,y,model)
%UPDATEBETA Summary of this function goes here
%   Detailed explanation goes here

%effectiveDim=find(model.alpha<10.^8);
initialBeta=model.beta;

func4fminunc=@(beta)func2minimize(X(:,model.effectiveDim),y,model,beta);
options = optimoptions('fminunc','Algorithm','trust-region',...
    'Display','off','SpecifyObjectiveGradient',true,'HessianFcn','objective','MaxIterations',10);
%options = optimset('Gradobj','on','Hessian','on','Display','iter');
[newBeta,funcVal,dummy,dummy,dummy,hessValue ] = fminunc(func4fminunc,initialBeta,options);
%[bestBeta4effectiveDim] = fminsearch(func4fminunc,initialBeta,options);
    

model.beta=newBeta;
model.covBeta=inv(hessValue);
end



function [funcValue gradValue hessValue] = func2minimize(X,y,model,betaAsArgument)
    y_1ofK=zeros(size(y,1),1+length(model.mu));
    for index_class=1:size(y_1ofK,2)
        y_1ofK(y==index_class,index_class)=1;
    end

    alpha=model.alpha;
    mu=model.mu;

    [P F f df]=calcPredictiveProbability_v3(X,betaAsArgument,mu);
    P(P<10.^(-150))=10.^(-150);
    funcValue=y_1ofK.*log(P);
    funcValue=sum(funcValue(:));
    funcValue=funcValue-0.5.*sum(alpha.*(betaAsArgument.^2));%add ARD prior effect
    funcValue=-funcValue;
    
    gradValue=zeros(length(alpha),1);
    for index_sample=1:size(y,1)
        if y(index_sample)==1
            dlogP_dbeta=f(index_sample,y(index_sample));
        else
            dlogP_dbeta=f(index_sample,y(index_sample))-f(index_sample,y(index_sample)-1);
        end
        dlogP_dbeta=dlogP_dbeta./P(index_sample,y(index_sample));
        gradValue=gradValue+dlogP_dbeta.*(-X(index_sample,:)');
    end
    %add ARD prior
    ARDterm=alpha.*betaAsArgument;
    gradValue=gradValue-ARDterm;
    gradValue=-gradValue;
                             
                              
                              
    hessValue=zeros(length(alpha),length(alpha));
    for index_sample=1:size(y,1)
        if y(index_sample)==1
            d2logP_dbetabeta=df(index_sample,y(index_sample))./P(index_sample,y(index_sample)) - ...
                (f(index_sample,y(index_sample))./P(index_sample,y(index_sample))).^2;
        else
            d2logP_dbetabeta=(df(index_sample,y(index_sample))-df(index_sample,y(index_sample)-1))./P(index_sample,y(index_sample)) - ...
                ((f(index_sample,y(index_sample))-f(index_sample,y(index_sample)-1))./P(index_sample,y(index_sample))).^2;
        end
        hessValue=hessValue+d2logP_dbetabeta.*((X(index_sample,:)')*X(index_sample,:));
    end
    %add ARD prior
    hessValue=hessValue-diag(alpha);
    hessValue=-hessValue;
end
