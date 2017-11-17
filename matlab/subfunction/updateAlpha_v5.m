function [ model ] = updateAlpha_v5(X,y,model)
%UPDATEALPHA Summary of this function goes here
%   Detailed explanation goes here
%At v5, if covBeta's elements are negative, replace them with 0.
alpha=model.alpha;
covBeta=model.covBeta;
beta=model.beta;

if 1%all(diag(covBeta)>0.00001)
dim2eliminate=[];
for index_alpha=1:length(alpha)
    if 1%covBeta(index_alpha,index_alpha) > 0
        %update rule like Yamashita et al
        alpha(index_alpha)=...
            (1-alpha(index_alpha).*max(covBeta(index_alpha,index_alpha),0))./((beta(index_alpha)).^2);
        %original update rule
        %alpha(index_alpha)=1./(max(covBeta(index_alpha,index_alpha),0)+((beta(index_alpha)).^2));
    end
    if alpha(index_alpha) > 10.^8
        dim2eliminate=[dim2eliminate index_alpha];
    end
    %Added ad v5.
    if alpha(index_alpha) < 0.01
        alpha(index_alpha)=0.01;
    end
end

if ~isempty(dim2eliminate)
    alpha(dim2eliminate)=[];
    beta(dim2eliminate)=[];
    covBeta(dim2eliminate,:)=[];
    covBeta(:,dim2eliminate)=[];
    model.effectiveDim(dim2eliminate)=[];
end

end

model.beta=beta;
model.alpha=alpha;
model.covBeta;

end

