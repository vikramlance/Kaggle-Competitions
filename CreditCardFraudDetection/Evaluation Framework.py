###Evaluation Framework for the Kaggle Credit Card Fraud Data Set
###By: Vikram, Andy 3/15/17

"""
Overview:
-These are European credit card transactions from 2013.
-We have 2 days of data
-We have 285,299 data points
-0.172% are true positives
-All features are hidden aside from time and amount taken.
-Hidden features are likely going to be deterministic on the model.
-Amount/Time may play a role but they are very generalized features.

Evaluation framework/objectives:
-Credit card fraud hurts credit firms.
-The firms must pay back what's stolen and will likely not catch the criminals.
-False positives are annoying for the customer, and have an associated negative cost
-However, the cost is acceptable, and exceptionally hard to quantify.
-False positives should be minimized, but should not be the focus.
-The focus should be on not tolerating False negatives.
-A false negative is an instantaneous cost to the firm.
-A false negative also cause non-quantifiable damage to the firm's intangible assets.

Evaluation solution:
-We need to use a method that utilizes the information above into a measureable metric.
-Precision-Recall methods and optimization will likely lead to the best solution.
-Reiterating the above, the PR method should first optimize the reduction of false negatives
-The method should then optimize for false positives.
-High recall (1) and a high F1 score should be optimized first.
-A high precision should be optimized second.

Tools:
-sklearn.metrics.recall_score
-sklearn.metrics.f1_score
-sklearn.metrics.average_precision_score
-sklearn.metrics.precision_score

Sample codes:

***from sklearn.metrics import recall_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
recall_score(y_true, y_pred, average='macro')  
recall_score(y_true, y_pred, average='micro')  
recall_score(y_true, y_pred, average='weighted')  
recall_score(y_true, y_pred, average=None)

***from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
f1_score(y_true, y_pred, average='macro')  
f1_score(y_true, y_pred, average='micro')  
f1_score(y_true, y_pred, average='weighted')  
f1_score(y_true, y_pred, average=None)

*import numpy as np
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
average_precision_score(y_true, y_scores) 

*from sklearn.metrics import precision_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
precision_score(y_true, y_pred, average='macro')  
precision_score(y_true, y_pred, average='micro')  
precision_score(y_true, y_pred, average='weighted')
precision_score(y_true, y_pred, average=None)  

"""
