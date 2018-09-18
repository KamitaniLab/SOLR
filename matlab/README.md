# MATLAB implementation of SOLR

Sparse ordinal logistic regression (SOLR) is a machine learning algorithm to predict ordinal variables.
This repository provides a MATLAB implementation of SOLR.
For the details of the algorithm, please see

Satake, E., Majima, K., Aoki, S.C., and Kamitani, Y. (2018). Sparse Ordinal Logistic Regression and Its Application to Brain Decoding. Frontiers in Neuroinformatics. <https://doi.org/10.3389/fninf.2018.00051>

## Core fucntions & demo codes

- SOLRtrain.m: A function for SOLR model training.
- SOLRpredict.m: A function for prediction.
- demoSOLR_20171227.m: Demo code of SOLR.
- demoComp_SOLRvsSMLRvsSLiR_20171227.m: Demo comparison across prediction methods.

## How to use

After adding “subfunction” folder to the MATLAB path, please type

```
>> model = SOLRtrain(feature, label);
>> predictedLabel = SOLRpredict(feature, model);
```

- `feature`: a matrix (# of training samples x # of dimensions).</li>
- `label`: a vector including label information (# of training samples x 1; elements must be natural numbers).</li>

`model` contains the parameters after model training, and SOLRpredict returns predictions based on it.

## Demonstration

demoComp_SOLRvsSMLRvsSLiR_20171227.m

<img src="figDemoComp_SOLRvsSMLRvsSLiR_20171227.png">
