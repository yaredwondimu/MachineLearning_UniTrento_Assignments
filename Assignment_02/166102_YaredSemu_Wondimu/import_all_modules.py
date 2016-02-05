'''
This file :
	- imports all the required packages and 
	- prepares the artificial dataset for training and testing purposes
'''

## Import classes to be used:
 # - the **GaussianNB**, classification algorithm (model)
 # - the **SVM**, classification algorithm (model)
 # - the **RandomeForestClassifier**, classification algorithm (model)
 # - the evaluation scheme **cross_validation**
 # - the artificial dataset
 # - performance evaluation metrics
 


#Cross Validation
from sklearn import cross_validation

#Classification Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Performance Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

#import matploit (scientific plotting library)
import matplotlib.pyplot as plot
