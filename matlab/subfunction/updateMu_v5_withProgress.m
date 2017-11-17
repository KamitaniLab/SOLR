function [ model ] = updateMu_v5_withProgress(X,y,model)
%UPDATEBETA Summary of this function goes here
%   Detailed explanation goes here

    initialMu=model.mu;
    func4fminunc=@(mu)func2minimize(X(:,model.effectiveDim),y,model,mu);
    matrix4constraint=-eye(length(model.mu),length(model.mu));
    matrix4constraint(1,1)=0;
    if length(model.mu) > 1
        for index_mu=2:length(model.mu)
            matrix4constraint(index_mu,index_mu-1)=1;
        end
    end
    matrix4constraint(1,:)=[];
   
    %options = optimset('Gradobj','on','Hessian','on','Display','off');
    %[bestMu] = fminunc(func4fminunc,initialMu,options);
    %options = optimset('Algorithm','active-set','Hessian','off','GradObj','on','Display','off');
    %options = optimset('Algorithm','trust-region-reflective','Hessian','on','GradObj','on','Display','off');
    
    hess4fmincon=@(mu,lambda)hess2minimize(X(:,model.effectiveDim),y,model,mu);
   
    
    options = optimoptions('fmincon','Algorithm','interior-point','Hessian','on','GradObj','on','Display','iter','HessianFcn',hess4fmincon,'MaxIterations',10);    
    [bestMu] = fmincon(func4fminunc,initialMu,matrix4constraint,zeros(size(matrix4constraint,1),1),...
        [],[],[],[],[],options);
    model.mu=bestMu;
end



function [funcValue gradValue hessValue] = func2minimize(X,y,model,muAsArgument)
    y_1ofK=zeros(size(y,1),1+length(model.mu));
    for index_class=1:size(y_1ofK,2)
        y_1ofK(y==index_class,index_class)=1;
    end

    alpha=model.alpha;
    %alpha=diag(alpha);
    mu=muAsArgument;

    beta=model.beta;


    [P F f df]=calcPredictiveProbability_v3(X,beta,mu);
    P(P<10.^(-150))=10.^(-150);
    funcValue=y_1ofK.*log(P);
    funcValue=sum(funcValue(:));
    funcValue=funcValue-0.5.*sum(alpha.*(beta.^2));%add ARD prior effect
    funcValue=-funcValue;
    
    gradValue=zeros(length(mu),1);
    for index_sample=1:size(y,1)
        dlogP_dmu=zeros(length(mu),1);
        %kokomade
        if y(index_sample)<=length(mu)
            dlogP_dmu(y(index_sample))=f(index_sample,y(index_sample))./P(index_sample,y(index_sample));
        end
        if y(index_sample)~=1
            dlogP_dmu(y(index_sample)-1)=-f(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample));
        end
        gradValue=gradValue+dlogP_dmu;
    end
    gradValue=-gradValue;
                             
                              
    %{                          
    hessValue=zeros(length(mu),length(mu));
    
    for index_sample=1:size(y,1)
        d2logP_dmumu=zeros(length(mu),length(mu));
        if y(index_sample)<=length(mu)
            d2logP_dmumu(y(index_sample),y(index_sample))=...
                df(index_sample,y(index_sample))./P(index_sample,y(index_sample))-...
                (f(index_sample,y(index_sample))./P(index_sample,y(index_sample))).^2;
        end
        
        if y(index_sample)~=1
            d2logP_dmumu(y(index_sample)-1,y(index_sample)-1)=...
                -df(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample))-...
                (-f(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample))).^2;
            if y(index_sample)<=length(mu)
                d2logP_dmumu(y(index_sample),y(index_sample)-1)=...
                    ((-f(index_sample,y(index_sample))).*(-f(index_sample,y(index_sample)-1)))./((P(index_sample,y(index_sample)).^2));
                d2logP_dmumu(y(index_sample)-1,y(index_sample))=d2logP_dmumu(y(index_sample),y(index_sample)-1);
            end
        end
        hessValue=hessValue+d2logP_dmumu;
    end
    hessValue=-hessValue;
    %}
end

function [hessValue] = hess2minimize(X,y,model,muAsArgument)
    y_1ofK=zeros(size(y,1),1+length(model.mu));
    for index_class=1:size(y_1ofK,2)
        y_1ofK(y==index_class,index_class)=1;
    end

    alpha=model.alpha;
    %alpha=diag(alpha);
    mu=muAsArgument;

    beta=model.beta;

    
    [P F f df]=calcPredictiveProbability_v3(X,beta,mu);
    %{
    P(P<10.^(-150))=10.^(-150);
    funcValue=y_1ofK.*log(P);
    funcValue=sum(funcValue(:));
    funcValue=funcValue-0.5.*sum(alpha.*(beta.^2));%add ARD prior effect
    funcValue=-funcValue;
    
    gradValue=zeros(length(mu),1);
    for index_sample=1:size(y,1)
        dlogP_dmu=zeros(length(mu),1);
        %kokomade
        if y(index_sample)<=length(mu)
            dlogP_dmu(y(index_sample))=f(index_sample,y(index_sample))./P(index_sample,y(index_sample));
        end
        if y(index_sample)~=1
            dlogP_dmu(y(index_sample)-1)=-f(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample));
        end
        gradValue=gradValue+dlogP_dmu;
    end
    gradValue=-gradValue;
                              
    %}                          
                              
    %{
    hessValue=zeros(length(mu),length(mu));   
    for index_sample=1:size(y,1)
        d2logP_dmumu=zeros(length(mu),length(mu));
        if y(index_sample)<=length(mu)
            d2logP_dmumu(y(index_sample),y(index_sample))=...
                df(index_sample,y(index_sample))./P(index_sample,y(index_sample))-...
                (f(index_sample,y(index_sample))./P(index_sample,y(index_sample))).^2;
        end
        
        if y(index_sample)~=1
            d2logP_dmumu(y(index_sample)-1,y(index_sample)-1)=...
                -df(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample))-...
                (-f(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample))).^2;
            if y(index_sample)<=length(mu)
                d2logP_dmumu(y(index_sample),y(index_sample)-1)=...
                    ((-f(index_sample,y(index_sample))).*(-f(index_sample,y(index_sample)-1)))./((P(index_sample,y(index_sample)).^2));
                d2logP_dmumu(y(index_sample)-1,y(index_sample))=d2logP_dmumu(y(index_sample),y(index_sample)-1);
            end
        end
        hessValue=hessValue+d2logP_dmumu;
    end
    hessValue=-hessValue;
    %}
    
    
    hessValue=sparse(length(mu),length(mu));
    for index_sample=1:size(y,1)
        d2logP_dmumu=sparse(length(mu),length(mu));
        if y(index_sample)<=length(mu)
            d2logP_dmumu(y(index_sample),y(index_sample))=...
                df(index_sample,y(index_sample))./P(index_sample,y(index_sample))-...
                (f(index_sample,y(index_sample))./P(index_sample,y(index_sample))).^2;
        end
        
        if y(index_sample)~=1
            d2logP_dmumu(y(index_sample)-1,y(index_sample)-1)=...
                -df(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample))-...
                (-f(index_sample,y(index_sample)-1)./P(index_sample,y(index_sample))).^2;
            if y(index_sample)<=length(mu)
                d2logP_dmumu(y(index_sample),y(index_sample)-1)=...
                    ((-f(index_sample,y(index_sample))).*(-f(index_sample,y(index_sample)-1)))./((P(index_sample,y(index_sample)).^2));
                d2logP_dmumu(y(index_sample)-1,y(index_sample))=d2logP_dmumu(y(index_sample),y(index_sample)-1);
            end
        end
        hessValue=hessValue+d2logP_dmumu;
    end
    hessValue=-hessValue;
    
end