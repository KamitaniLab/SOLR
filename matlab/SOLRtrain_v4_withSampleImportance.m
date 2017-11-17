function [ model ] = SOLRtrain_v4_withSampleImportance(feature,label,weight4sample)
%SOLRTRAIN Summary of this function goes here
%   Detailed explanation goes here
%[ model ] = SOLRtrain(feature,label)
%feature: a matrix (# of training samples x # of dimensions).
%label: a vector including label information (# of training samples x 1; elements must be natural numbers).

%At v3, calculation was conducted while focusing on effective dimensions
%more for speeding up.
%At v4, the # of subiterations in updateBeta was reduced more for speed up.

%At v4_withSampleImportance, to adjust SOLR for imbalanced classes (classification with imbalanced numbers of samples across classes),
%weight4sample wad added as the 3rd input.

%set initial parameter values.
model.beta=zeros(size(feature,2),1);
model.mu=linspace(-1,1,max(label)-1);
model.alpha=ones(size(feature,2),1);
model.effectiveDim=1:size(feature,2);

%variational bayse iterations.
for index_iteration=1:100
    if index_iteration < 3
        numSubiterations=10;
    else
        numSubiterations=1;
    end
    for index_subiteration=1:numSubiterations
        model=updateBeta_v4_withSampleImportance(feature,label,model,weight4sample);
        model=updateMu_v3_withSampleImportance(feature,label,model,weight4sample);
    end
    model=updateAlpha_v3(feature,label,model);
    
    if rem(index_iteration,1)==0
        display(['Iteration: ' num2str(index_iteration) ', # of effective dimensions:' num2str(length(model.beta))])
    end
end

end

