# Core fucntions & demo codes
・SOLRtrain.m: A function for SOLR model training.
・SOLRpredict.m: A function for prediction.
・demoSOLR_20171227.m: Demo code of SOLR.
・demoComp_SOLRvsSLRvsSLiR_20171227.m: Demo comparison across prediction methods.

# How to use
After adding “subfunction” folder to the MATLAB path, please type

model = SOLRtrain(feature,label);
predicted = SOLRpredict(feature,model);
.
Here, 
・feature: a matrix (# of training samples x # of dimensions).
・label: a vector including label information (# of training samples x 1; elements must be natural numbers).
“model” contains the parameters after model training, 
and SOLRpredict returns predictions based on it.

#Demonstration
demoComp_SOLRvsSLRvsSLiR_20171227.m
[ここにデモコードででてくるfigureを貼り付ける。]
