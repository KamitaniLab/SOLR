function [ model ] = updateAlpha_forL2OLR_v2(X,y,model)
%UPDATEALPHA Summary of this function goes here
%   Detailed explanation goes here
%If covBeta's diagonal elemtns are not positive, fix it.

covBeta=model.covBeta;
beta=model.beta;
varBeta=diag(covBeta);
varBeta(varBeta<0)=0;

alpha=length(beta)./(sum(varBeta)+sum((beta).^2));

model.alpha=alpha.*ones(length(beta),1);

end

