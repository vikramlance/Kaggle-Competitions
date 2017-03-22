###Evaluation Framework for the Kaggle Credit Card Fraud Data Set
###By: Vikram, Andy 3/15/17
###Python version: 3.5

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
### Interesting AUP ROC Curve link:
### http://stats.stackexchange.com/questions/21592/optimising-for-precision-recall-curves-under-class-imbalance

#import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle


from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


#Omit for now
"""
#data import ***WILL NEED UPDATED
data = loadData
x = xData
y = yData

# setup plot details
#colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
#lw = 2

# binarize the output
y = label_binarize(y, classes=[0, 1])
n_classes = y.shape[1]


#train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.666)

# Run classifier
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
"""



#Precision recall curve 
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")


# Plot Precision-Recall curve (this is if we want to use plt)
"""
plt.clf()
plt.plot(recall[0], precision[0], lw=lw, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.show()

# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()
"""
