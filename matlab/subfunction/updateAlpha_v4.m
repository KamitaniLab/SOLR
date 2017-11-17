function [ model ] = updateAlpha_v4(X,y,model)
%UPDATEALPHA Summary of this function goes here
%   Detailed explanation goes here

alpha=model.alpha;
covBeta=model.covBeta;
beta=model.beta;

dim2eliminate=[];
for index_alpha=1:length(alpha)
    %update rule like Yamashita et al
    %alpha(index_alpha)=...
    %    (1-alpha(index_alpha).*covBeta(index_alpha,index_alpha))./((beta(index_alpha)).^2);
    %original update rule
    alpha(index_alpha)=1./(covBeta(index_alpha,index_alpha)+((beta(index_alpha)).^2));
    if alpha(index_alpha) > 10.^8
        dim2eliminate=[dim2eliminate index_alpha];
    end
end

if ~isempty(dim2eliminate)
    alpha(dim2eliminate)=[];
    beta(dim2eliminate)=[];
    covBeta(dim2eliminate,:)=[];
    covBeta(:,dim2eliminate)=[];
    model.effectiveDim(dim2eliminate)=[];
end

model.beta=beta;
model.alpha=alpha;
model.covBeta;

end

