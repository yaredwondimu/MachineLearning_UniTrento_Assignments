## Import classes to be used:
 # - the **SMM**, classification algorithm (model) 
 # - the evaluation scheme **cross_validation**
 # - the artificial dataset
 # - performance evaluation metrics
 

#Numpy Library 
import numpy as np
 
#Cross Validation
from sklearn import cross_validation
from sklearn import datasets

#Classification Algorithm
from sklearn.svm import SVC

# Performance Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

#import matploit (scientific plotting library)
import matplotlib.pyplot as plot

#import utitlity modules
from utilities import svm_rbf
from utilities import average_performance_metric
from utilities import write_to_file

#GridSearchCV
from sklearn.grid_search import GridSearchCV


## Load and generate the artificial dataset from sklearn's artificial data generator

# Properties of the artificial dataset
  # - num_samples = 1000
  # - num_features = 10
 
# Load and generate the dataset
dataset = datasets.make_classification(n_samples=1000,n_features=10, 
									   n_informative=2,n_redundant=2,
									   n_repeated=0, n_classes=2)

## Lists for storing the performance evaluation schemes 
## lists for holding the accuracy, f1 and auc score in each fold for GaussianNB 
accuracy_SVM=[]
f1_SVM=[]
auc_roc_SVM=[]

#10-fold cross-validation
#Split the dataset using 10-fold cross validation 
kf = cross_validation.KFold(len(dataset[0]), n_folds=10, shuffle=False,random_state=None)


## Iterate through each of the fold (0-9), in each iteration; 
 # - Train the model 
 # - Test the model 
 # - Compute the Performance of the learned model
 # - Store the performance computed values for later analysis

for train_index, test_index in kf:			
	features_train = dataset[0][train_index]
	labels_train = dataset[1][train_index]
	
	features_test = dataset[0][test_index]	
	labels_test =  dataset[1][test_index]
	
	#Possible values of C
	C = [1e-02, 1e-01, 1e00, 1e01, 1e02]
	bestC = None
	innerscore=[]
	
	for c in C:
		ikf= cross_validation.KFold(len(features_train), n_folds=5, shuffle=True,random_state=5678)
		innerf1=[]
		for t_index, v_index in ikf:
			X_t, X_v = features_train[t_index], features_train[v_index]
			y_t, y_v = labels_train[t_index], labels_train[v_index]

			ipred=svm_rbf(X_t,y_t,X_v,c)
			
			#save the f1 score for inner cross validation
			innerf1.append(f1_score(y_v,ipred))
			
		#compute the average
		innerscore.append(sum(innerf1)/len(innerf1))
	
	#pick C that give the best f1	
	bestC=C[np.argmax(innerscore)]
	print ""
	#predict the labels for the test set using best c parameter
	labels_predictions =svm_rbf(features_train,labels_train,features_test,bestC)   
	
	#calculating the performance
	svm_accuracy = accuracy_score(labels_test,labels_predictions)
	svm_f1 = f1_score(labels_test,labels_predictions)
	svm_auc_roc = roc_auc_score(labels_test,labels_predictions)
	svm_ave_preci = average_precision_score(labels_test,labels_predictions)
  
	#appending the results to the list for computing the average performance of SVM
	accuracy_SVM.append(svm_accuracy)
	f1_SVM.append(svm_f1)
	auc_roc_SVM.append(svm_auc_roc)
	
#Record per fold performance results in a file
write_to_file("svm",accuracy_SVM,f1_SVM,auc_roc_SVM)

#compute Average performance measure For GaussianNaiveBayes
print "\n##Average performance measure for SVM with a RBF kernel##\n"
print "\tAverage Accuracy: = "+ average_performance_metric(accuracy_SVM)
print "\tAverage F1 Score: = "+ average_performance_metric(f1_SVM)
print "\tAverage AUC ROC:  = "+ average_performance_metric(auc_roc_SVM)
print

