'''
This file ....
'''
from import_all_modules import *

## Parameters
## @features_train : Training data representing the features
## @labels_train : Labels corresponding to the features_train vector
## @features_test : Testing data representing the features
## @kernel : Kernel type of the SVM classification model (possible kernels : rbf,linear,sigmod)
## @c : for the SVM classification model for the SVM classification model

#Returns
## The predicted labels for the corresponding features_test data
def svm_rbf(features_train,labels_train,features_test,c):
	svm_rbf_clf = SVC (kernel="rbf",C=c,class_weight='balanced')
	svm_rbf_clf.fit(features_train,labels_train)
	return svm_rbf_clf.predict(features_test)

## Parameters
## @features_train : Training data representing the features
## @labels_train : Labels corresponding to the features_train vector
## @features_test : Testing data representing the features
## @n_estimators: Number of estimators 
## @criterion: 
## @estimator : for the SVM classification model for the SVM classification model

#Returns
## The predicted labels for the corresponding features_test data
def random_forest_gini(features_train,labels_train,features_test,estimator):
	rnd_frst_gini_clf = RandomForestClassifier(n_estimators=estimator,criterion="gini")
	rnd_frst_gini_clf.fit(features_train,labels_train,)
	return rnd_frst_gini_clf.predict(features_test)
	

## Parameters
## @features_train : Training data representing the features
## @labels_train : Labels corresponding to the features_train vector
## @features_test : Testing data representing the features
## @kernel : Kernel type of the SVM classification model (possible kernels : rbf,linear,sigmod)
## @c : for the SVM classification model for the SVM classification model

#Returns
## The predicted labels for the corresponding features_test data
def gaussian_naive_bayes(features_train,labels_train,features_test):
	gnb_clf = GaussianNB()
	gnb_clf.fit(features_train,labels_train)
	return gnb_clf.predict(features_test)

def average_performance_metric(performance_metrics):
	return str( "%.2f" % (sum(performance_metrics)/len(performance_metrics)))
	
## Write Performance metrics results to a file, per fold
def write_to_file(clf_algorithm,accuracy,f1,auc_roc):
	if clf_algorithm == 'gnb':
		file_stream = open("performance_results_gnb.dat", "w")
	elif clf_algorithm == 'svm':
		file_stream = open("performance_results_svm.dat", "w")
	else:
		file_stream = open("performance_results_rndfrst.dat", "w")
		
	file_stream.write("fn, acc, f1, auc"+'\n')
	file_stream.write("-------------------"+'\n')

	for i in range(10):	
		#Write the results to a file			
		file_stream.write(str(i) +', '+ str("%.2f" % accuracy[i]) +', '+ str("%.2f" % f1[i])+ ', '+ str("%.2f" % auc_roc[i])+'\n')	
	file_stream.close()	
