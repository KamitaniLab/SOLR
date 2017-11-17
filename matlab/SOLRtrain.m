function [ model ] = SOLRtrain(feature,label)
%SOLRTRAIN Summary of this function goes here
%   Detailed explanation goes here
%[ model ] = SOLRtrain(feature,label)
%feature: a matrix (# of training samples x # of dimensions).
%label: a vector including label information (# of training samples x 1; elements must be natural numbers).


%set initial parameter values.
model.beta=zeros(size(feature,2),1);
model.mu=linspace(-1,1,max(label)-1);
model.alpha=ones(size(feature,2),1);

%variational bayse iterations.
for index_iteration=1:100
    for index_subiteration=1:1
        model=updateBeta(feature,label,model);
        model=updateMu(feature,label,model);
    end
    model=updateAlpha(feature,label,model);
    
    if rem(index_iteration,1)==0
        display(['Iteration: ' num2str(index_iteration) ', # of effective dimensions:' num2str(sum(model.beta~=0))])
    end
end

end

