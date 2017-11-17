function [ probability F f df] = calcPredictiveProbability_v3(X,beta,mu)
%PREDICTIVEPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

%At v3, the results were the same, but the code was changed to speed up.

activationValue=X*beta;
probability=zeros(size(X,1),length(mu)+1);
F =zeros(size(X,1),length(mu)+1);
f =zeros(size(X,1),length(mu)+1);
df=zeros(size(X,1),length(mu)+1);

probability(:,1)=logsig(mu(1)-activationValue);
if length(mu) > 1
    for index_class=2:length(mu)
        probability(:,index_class)=...
            logsig(mu(index_class)-activationValue)-logsig(mu(index_class-1)-activationValue);
    end
end
probability(:,length(mu)+1)=1-logsig(mu(end)-activationValue);




for index_class=1:length(mu)
    F(:,index_class)=logsig(mu(index_class)-activationValue);
    f(:,index_class)=pdf('logistic',mu(index_class)-activationValue);
    df(:,index_class)=-0.25.*tanh(mu(index_class)-activationValue)./...
        ((cosh(mu(index_class)-activationValue)).^2);
end
F(:,length(mu)+1)=1;
f(:,length(mu)+1)=0;
df(:,length(mu)+1)=0;