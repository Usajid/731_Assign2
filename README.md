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

The classifiation accuracy results are stated below:

```
Multinomial Naive Bayes Accuracy: 46%

Linear SVC (Without any Kernel) Accuracy: 92.2%

Logistic Regression Accuracy: 54.1%

Random Forest Accuracy: 93.7%

```

After feature transformation and data cleaning process, we did a training/testing split of around (75%/25%) respectively from total 39,826 samples.

For detailed analysis and time consumption details, please see the notebook (notebooks/shakespeare.ipynb)

# Conclusion
First, we pre-processed and balanced the given Shakespeare dataset. Then, we analyzed four major machine learning classifiers to classify correct Players label. Out of four tested classifiers, Random Forest based classifier outperformed other methods with best accuracy of 93.7%. One reason of Random Forest, being the best, is their property of ensembleness as final classifiation score is not based on just one Decision Tree, but from several Decision Trees using majority voting mechanism. They are also less prone to over-fitting/high-variance as each tree learns to predict differently.
