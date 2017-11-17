function [ model ] = L2SOLRtrain_v4(feature,label)
%SOLRTRAIN Summary of this function goes here
%   Detailed explanation goes here
%[ model ] = SOLRtrain(feature,label)
%feature: a matrix (# of training samples x # of dimensions).
%label: a vector including label information (# of training samples x 1; elements must be natural numbers).

%At v4, SOLRtrain_v5.m was modified to L2-constraint.

%set initial parameter values.
model.beta=zeros(size(feature,2),1);
model.mu=linspace(-1,1,max(label)-1);
model.alpha=ones(size(feature,2),1);
model.effectiveDim=1:size(feature,2);

%variational bayse iterations.
for index_iteration=1:10
    if index_iteration < 3
        numSubiterations=10;
    else
        numSubiterations=1;
    end
    for index_subiteration=1:numSubiterations        
        model=updateBeta_v4(feature,label,model);
        model=updateMu_v5(feature,label,model);
    end
    model=updateAlpha_forL2OLR_v2(feature,label,model);
    
    if rem(index_iteration,1)==0
        display(['Iteration: ' num2str(index_iteration) ', alpha:' num2str(model.alpha(1))])
    end
end

end

