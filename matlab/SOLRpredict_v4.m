function [ predictedLabel predictiveProbability ] = SOLRpredict_v4(feature,model)
%SOLRPREDICT Summary of this function goes here
%   Detailed explanation goes here
%[ predictedLabel predictiveProbability ] = SOLRpredict(feature,model)
%feature: a matrix including input feature values (# of samples x dimensions).
%model: a structure obtained by SOLRtrain.m
%

predictiveProbability=calcPredictiveProbability(feature(:,model.effectiveDim),model.beta,model.mu);
[dummy predictedLabel]=max(predictiveProbability,[],2);
    
end

