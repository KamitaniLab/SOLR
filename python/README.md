# Python implementation of SOLR

Sparse ordinal logistic regression (SOLR) is a machine learning algorithm to predict ordinal variables.
This repository provides a Python implementation of SOLR.
For the details of the algorithm, please see

Satake, E., Majima, K., Aoki, S.C., and Kamitani, Y. (2018). Sparse Ordinal Logistic Regression and Its Application to Brain Decoding. Frontiers in Neuroinformatics. <https://doi.org/10.3389/fninf.2018.00051>

## Core fucntions & demo codes

- SOLR.py: A function for model training and prediction.
- demoSOLR_20171227.py: Demo code of SOLR.
- demoComp_SOLRvsSMLRvsSLiR_20171227.py: Demo comparison across prediction methods.

## How to use

Please type

```
>> import SOLR
>> solr = SOLR.SOLR()
>> solr.fit(feature, label)
>> predicted_label = solr.predict(feature, model)
```

- `feature`: a numpy array (# of training samples x # of dimensions).</li>
- `label`: a numpy array including label information (# of training samples x 1; elements must be nonnegative integers).

The output array (`predicted_label`) contains predictions by SOLR.

## Demonstration

demoComp_SOLRvsSMLRvsSLiR_20171227.py

<img src="figDemoComp_SOLRvsSMLRvsSLiR_20171227.png">
