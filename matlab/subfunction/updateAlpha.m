function [ model ] = updateAlpha(X,y,model)
%UPDATEALPHA Summary of this function goes here
%   Detailed explanation goes here

alpha=model.alpha;
covBeta=model.covBeta;
beta=model.beta;
for index_alpha=1:length(alpha)
    if alpha(index_alpha) > 10.^8
        model.beta(index_alpha)=0;
    else
        %update rule like Yamashita et al
        alpha(index_alpha)=...
            (1-alpha(index_alpha).*covBeta(index_alpha,index_alpha))./((beta(index_alpha)).^2);
        %original update rule
        %alpha(index_alpha)=1./(betaCov(index_alpha,index_alpha)+((beta(index_alpha)).^2));
    end
end
model.beta=beta;
model.alpha=alpha;

end

