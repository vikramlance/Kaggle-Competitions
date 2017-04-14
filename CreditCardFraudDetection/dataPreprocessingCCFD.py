#I'm going to go ahead and import the libraries that I think I'll be using.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
%matplotlib inline

#Lets get the data and take a look.
data = pd.read_csv("ccfdRaw")
data.head(10)

#so I like the idea of getting rid of time and instantly normalizing the amount feature.
data['nAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))
data=data.drop(['Time', 'Amount'],axis=1)
data.head(10)

#Great now we have all our features in a normalized format.
#But how do we continue our feature engineering if we don't know meaning of the data?

#This won't finish running, oh well I tried.

###I WANTED: to see if eliminating insiginificant features can increase the models predictability...
###Maybe there's a better way.


"""
from sklearn.svm import SVC
from sklearn.model_selection import StraitfiedKFold
from sklearn.feature_selection import RFECV

y=data['Class']
X=data.drop('Class',1)

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step = 1, cv = StratifiedKFold(2))
rfecv.fit(X,y)


print("Optimal number of features : %d" % rfecv.n_features_)


"""

#Our data set is highly skewed. There are only ~500 fraud cases out of ~300,000 total cases.
#The training data needs to be balanced.

fraudCases = len(data[data.Class==1]) #fraud = 492 pieces
fraudIndices = np.array(data[data.Class==1].index) #tells us the row in array format. For Fraud

nonFraud = len(data[data.Class==0]) #non-fraud = 284,315 pieces
nonFraudIndices = np.array(data[data.Class==0].index) #tells us the row in array format. For non-fraud

#Great but we need an undersampled version of the non-fraud indices
randNonFraudIndices = np.random.choice(nonFraudIndices, fraudCases, replace=False) #Generates 492 random non-fraud case indices for us.

#Combining and pulling data
UnderSampledIndices = np.concatenate([randNonFraudIndices, fraudIndices]) #Combines the indices.
underSampled_data = data.iloc[UnderSampledIndices,:] #Grabs us the data that our undersampled indices points to

#For training and testing large data:
x=data.ix[:,data.columns != 'Class']
y=data.ix[:,data.columns == 'Class']
#Creates 2 data sets, one with class and indices, (y), and one with attributes and no class, (x)

#Undersampled x,y:
xU=underSampled_data.ix[:,underSampled_data.columns != 'Class'] #x from above just undersampled
yU=underSampled_data.ix[:,underSampled_data.columns == 'Class'] #y from above just undersampled

#It's important to note that undersampling will slightly skew the data.
#The mean, the min, and the max will change.
#It might be a good idea to re-scale the undersampled data with min/max to improve accuracy.
#That is going to cause some form of innacuracy to develop. 
#Just a good note, I'm not sure how to deal with it yet.

#The data is pre-processed now. Undersampled data will be trained/tested with entire data set.
