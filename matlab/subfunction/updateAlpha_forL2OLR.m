function [ model ] = updateAlpha_forL2OLR(X,y,model)
%UPDATEALPHA Summary of this function goes here
%   Detailed explanation goes here

covBeta=model.covBeta;
beta=model.beta;

alpha=length(beta)./(sum(diag(covBeta))+sum((beta).^2));

model.alpha=alpha.*ones(length(beta),1);

end

