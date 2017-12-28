# Python implementation of SOLR
Sparse ordinal logistic regression (SOLR) is a machine learning algorithm to predict ordinal variables. <br>
This repository provides a Python implementation of SOLR. <br> <br>
For the details of the algorithm, please see <br> <br>
Sparse ordinal logistic regression and its application to brain decoding. <br>
Emi Satake, Kei Majima, Shuntaro Aoki, Yukiyasu Kamitani. 2017. <br>
https://www.biorxiv.org/content/early/2017/12/22/238758.

## Core fucntions & demo codes
<ul>
  <li>SOLR.py: A function for model training and prediction.</li>
  <li>demoSOLR_20171227.py: Demo code of SOLR.</li>
  <li>demoComp_SOLRvsSLRvsSLiR_20171227.py: Demo comparison across prediction methods.</li>
</ul>

## How to use
Please type
```
>> import SOLR
>> solr = SOLR.SOLR()
>> solr.fit(feature,label)
>> predictedLabel = solr.predict(feature,model)
```
. Here, 
<ul>
  <li>feature: a numpy array (# of training samples x # of dimensions).</li>
  <li>label: a numpy array including label information (# of training samples x 1; elements must be nonnegative integers).</li>
</ul>
The output array (predictedLabel) contains predictions by SOLR.

## Demonstration
demoComp_SOLRvsSLRvsSLiR_20171227.py <br>
[Output figure from demo code]
<img src="figure_1.png">
