function [ model ] = L2OLRtrain_v3(feature,label)
%SOLRTRAIN Summary of this function goes here
%   Detailed explanation goes here
%[ model ] = SOLRtrain(feature,label)
%feature: a matrix (# of training samples x # of dimensions).
%label: a vector including label information (# of training samples x 1; elements must be natural numbers).

%At v3 of L2-OLR, alpha was fixed to 1.

%set initial parameter values.
model.beta=zeros(size(feature,2),1);
model.mu=linspace(-1,1,max(label)-1);
model.alpha=ones(size(feature,2),1);
model.effectiveDim=1:size(feature,2);

%variational bayse iterations.
for index_iteration=1:30    
    model=updateBeta_v4_withProgress(feature,label,model);
    model=updateMu_v5_withProgress(feature,label,model);

    if rem(index_iteration,1)==0
        display(['Iteration: ' num2str(index_iteration) ])
    end
end

end

