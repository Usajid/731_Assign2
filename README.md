# EECS-731: Assignment 2 (To be or not to be)
---
Our jupyter notebook for this assignment can be found in the 'notebooks' directory (notebooks/shakespeare.ipynb).


The soure code can be found in the src directory (src/shakespeare.py)

I used the four major classifiers, for the given classification task, as following:
- Multinomial Naive Bayes
- Random Forest Classifier
- Linear Support Vector Classifier (Without Kernel)
- Logistic Regression

The PDF version of notebook or report can be found in the reports directory (reports/shakespeare.pdf)


# UPDATE
After the discussion in the class today (09/24/2018), I got the idea to train
my classifiers on the ENTIRE dataset instead of training on one set and testing
on the other unseen set. With that, I get best case **accuracy of ~95% with
Random Forest Classifier** on the filtered (balanced) dataset.
Linear SVC was a close second with **~93%** accuracy.

# UPDATE-2
Just finished executing my program on full dataset in EECS cycle servers.
On full dataset, accuracy of **Random Forest is ~96%** and also orders of
magnitude faster than the other classifiers (maybe due to the parallelization
i.e., I used 24-jobs for this classifier). The detailed results are stated
below:

```
## Shakespear Dataset Dimensions:  (111396, 6)
## Shakespear Dataset Dimensions (Without Missing Values):  (105152, 7)
---
## Feaature Matrix Shape:  (105152, 12710)
## Testing Matrix Shape :  (26288, 12710) (26288,)
---
## Classifying with Random Forest Classifer [RFC]...
[RFC] Score: 0.961
---
## Classifying with Logistic Regression Classifer [LRC]...
[LRC] Score: 0.408
---
## Classifying with Linear SV-Classifer [SVC]...
[SVC] Score: 0.877
---
## ==================== TIMING STATS (in seconds)
Data Reading                   : 0.18
Feature Extraction             : 0.04
Feature Transformation         : 3.71
Test Preparation               : 3.66
Random Forest                  : 44.99
Logistic Regression            : 799.13
Linear SVC                     : 509.69
```
