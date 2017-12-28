# Core fucntions & demo codes
<ul>
  <li>SOLRtrain.m: A function for SOLR model training.</li>
  <li>SOLRpredict.m: A function for prediction.</li>
  <li>demoSOLR_20171227.m: Demo code of SOLR.</li>
  <li>demoComp_SOLRvsSLRvsSLiR_20171227.m: Demo comparison across prediction methods.</li>
</ul>

# How to use
After adding “subfunction” folder to the MATLAB path, please type
```
>> model = SOLRtrain(feature,label);
>> predicted = SOLRpredict(feature,model);
```
. Here, 
<ul>
  <li>feature: a matrix (# of training samples x # of dimensions).</li>
  <li>label: a vector including label information (# of training samples x 1; elements must be natural numbers).</li>
</ul>
“model” contains the parameters after model training, 
and SOLRpredict returns predictions based on it.

# Demonstration
demoComp_SOLRvsSLRvsSLiR_20171227.m <br>
[Output figure from demo code]
