# EECS-731: Assignment 2 (To be or not to be)
---
Our jupyter notebook for this assignment can be found in the 'notebooks' directory.

- [./notebooks/nb2-shakespeare.ipynb](Jupyter Notebook)

I have also separated the code from the notebook into a standalone python file;
so that I may experiment with the entire dataset on the EECS cycle servers (My
personal machines were not capable enough for dealing with the dataset in its
entirety). These can be seen here:

- [./src/shakespeare.py](Simple test with entire dataset)
- [./src/shakespeare2.py](Advanced test with entire dataset using custom TfIdfVectorization)

I used the following classifiers:
- Random Forest
- Linear SVC
- Logistic Regression

I used the following features:
- Player Line
- Play

I combined these two features into a single one for each dataline and calculated TfIdfVectors for them for training the classifers.

I also tried Multinomial Naive Bayes but it was excruciatingly slow. Among all
the classifers, Logistic regression showed best accuracy (~23%) for the entire
dataset and ~13% for the filtered subset of data.

The PDF version of notebook can be seen here:
- [./src/nb2-shakespeare.pdf](Report)

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
