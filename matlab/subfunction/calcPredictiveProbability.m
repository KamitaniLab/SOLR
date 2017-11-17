function [ probability F f df] = calcPredictiveProbability(X,beta,mu)
%PREDICTIVEPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

activationValue=X*beta;
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