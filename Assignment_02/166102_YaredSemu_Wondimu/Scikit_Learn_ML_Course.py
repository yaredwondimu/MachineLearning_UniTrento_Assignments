
# coding: utf-8

# In[15]:

import sklearn
import numpy as np

#import modules
from sklearn import datasets

#Cross Validation
from sklearn import cross_validation

#Classfication Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Performance Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# In[10]:

# Load and generate the dataset
dataset = datasets.make_classification(n_samples=1600,
                                       n_features=16, n_informative=2, 
                                       n_redundant=2, n_repeated=0, n_classes=2)


# In[17]:

#SVM with RBF kernel
def rbf_svm(X_train,y_train,X_test,c):
    clf= SVC(C=c,kernel='rbf',class_weight='auto')
    clf.fit(X_train,y_train)
    return clf.predict(X_test)

#RandomForest with Gini 
def gini_RF(X_train,y_train,X_test,estimator):
    clfRF = RandomForestClassifier(n_estimators=estimator, criterion='gini', random_state=None)
    clfRF.fit(X_train, y_train)
    predRF= clfRF.predict(X_test)
    return predRF


# In[18]:

#lists for holding the accuracy, f1 and auc score in each fold for GaussianNB 
accuracyGNB=[]
f1GNB=[]
auc_rocGNB=[]

#lists for holding the accuracy, f1 and auc score in each fold for SVM 
accuracy_SVM=[]
f1_SVM=[]
auc_roc_SVM=[]

#lists for holding the accuracy, f1 and auc score in each fold for RF 
accuracy_RF=[]
f1_RF=[]
auc_roc_RF=[]
avprecisionRF=[]


# In[ ]:

#Split the dataset using 10-fold cross validation 
kf= cross_validation.KFold(len(dataset[0]), n_folds=10, shuffle=False,random_state=None)

for train_index, test_index in kf:
    X_train, X_test = dataset[0][train_index], dataset[0][test_index]
    y_train, y_test = dataset[1][train_index], dataset[1][test_index]

    #Debug output
        #print X_train
        #print X_test
        #print y_test
        #print y_train

    # Initialize the GaussianNB algorithm, #Train the GNB model and #Test the model generalized using GNB
    clf = GaussianNB()        
    clf.fit(X_train, y_train)        
    pred= clf.predict(X_test)
    
    #Inspect the data structures
        #print pred
        #print y_test
    
    # Evaluate the performance of the learned model
    GAccuracy= accuracy_score(y_test, pred)
    Gf1= f1_score(y_test, pred)
    Gauc= roc_auc_score(y_test,pred)
    
    #Debggging output
        #print GAccuracy
        #print Gf1
        #print Gauc

    #SVM algorithm
    bestC=None
    Cvalues=[1e-2,1e-1,1e0,1e1,1e2]
    innerscore=[]
    for C in Cvalues:
        ikf= cross_validation.KFold(len(X_train), n_folds=5, shuffle=True,random_state=5678)
        innerf1=[]
        for t_index, v_index in ikf:
            X_t, X_v = X_train[t_index], X_train[v_index]
            y_t, y_v = y_train[t_index], y_train[v_index]

            ipred=rbf_svm(X_t,y_t,X_v,C)
        #save the f1 score for inner cross validation
        innerf1.append(metrics.f1_score(y_v,ipred))
        #compute the average
        innerscore.append(sum(innerf1)/len(innerf1))
    #pick C that give the best f1
    bestC=Cvalues[np.argmax(innerscore)]
    #predict the labels for the test set using best c parameter
    predSVM=rbf_svm(X_train,y_train,X_test,bestC)


# In[ ]:



