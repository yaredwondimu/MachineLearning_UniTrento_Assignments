
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score



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

# creating a classiffcation dataset with 1600 samples, made up of 16 features. The data set has two classes

dataset = datasets.make_classification(n_samples=1600,
			 n_features=16, n_informative=2, 
	                 n_redundant=2, n_repeated=0, n_classes=2)

#Spliting the dataset using 10-fold cross validation 
kf= cross_validation.KFold(len(dataset[0]), n_folds=10, shuffle=False,random_state=None)

for train_index, test_index in kf:
	X_train, X_test = dataset[0][train_index], dataset[0][test_index]
	y_train, y_test = dataset[1][train_index], dataset[1][test_index]

	# GaussianNB algorithm
	clf = GaussianNB()
	clf.fit(X_train, y_train)
	pred= clf.predict(X_test)

	#calculating the performance
	GAccuracy= accuracy_score(y_test, pred)
	Gf1= f1_score(y_test, pred)
	Gauc= sklearn.metrics.roc_auc_score(y_test,pred)

	#appending the results to the list for computing the average performace of GaussianNB
	accuracyGNB.append(GAccuracy)
	f1GNB.append(Gf1)
	auc_rocGNB.append(Gauc)

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

	#calculating the performance
	SVMaccuracy = metrics.accuracy_score(y_test,predSVM)
	SVMf1=metrics.f1_score(y_test,predSVM)
	SVMauc=metrics.roc_auc_score(y_test,predSVM)
	SVMavepreci=average_precision_score(y_test,predSVM)
  
	#appending the results to the list for computing the average performace of SVM
	accuracy_SVM.append(SVMaccuracy)
	f1_SVM.append(SVMf1)
	auc_roc_SVM.append(SVMauc)

	
	#Random Forest
	
	best_n_estimator=None
	n_estimatorsvalues=[10,100,1000]
	RFinnerscore=[]
	for e in n_estimatorsvalues:
		ikf= cross_validation.KFold(len(X_train), n_folds=5, shuffle=True,random_state=4321)
		innerf2=[]
		for t_index, v_index in ikf:
			X_t, X_v = X_train[t_index], X_train[v_index]
			y_t, y_v = y_train[t_index], y_train[v_index]

			ipred=gini_RF(X_t,y_t,X_v,e)
			#save the f1 score for inner cross validation
			innerf2.append(metrics.f1_score(y_v,ipred))
		#compute the average
		RFinnerscore.append(sum(innerf2)/len(innerf2))

	best_n_estimator=n_estimatorsvalues[np.argmax(RFinnerscore)]
	predRF=gini_RF(X_train,y_train,X_test,best_n_estimator)

	#calculating the performance
	RFaccuarcy = metrics.accuracy_score(y_test,predRF)
	RFf1= metrics.f1_score(y_test,predRF)
	RFauc = metrics.roc_auc_score(y_test,predRF)
	
	#appending the results to the list for computing the average performace of RF
	accuracy_RF.append(RFaccuarcy)
	f1_RF.append(RFf1)
	auc_roc_RF.append(RFauc)

	
print "fold\tGNBAccuracy\tGNBF1\tGNBAUC\tSVMAccuracy\tSVMF1\tSVNAUC\tRFAccuracy\tRFF1\tRFAUC "
for i in range(10):
	print str(i+1)+"\t" +str (accuracyGNB[i])+"\t"+ str(f1GNB[i])+"\t" + str(auc_rocGNB[i])+"\t" +str (accuracy_SVM[i]) +"\t"+ str(f1_SVM[i])+"\t" + str(auc_roc_SVM[i])+"\t"+str (accuracy_RF[i])+"\t"+ str(f1_RF[i]) +"\t"+ str(auc_roc_RF[i])+"\n"

#compute Average performance measure For GaussianNB"
print "\n\tAverage performance measure For GaussianNB"
print "\tAverage Accuracy: = "+ str(sum(accuracyGNB)/len(accuracyGNB))
print "\tAverage F1 Score: = "+ str ( sum(f1GNB)/len(f1GNB))
print "\tAverage AUC ROC:  = "+ str (sum(auc_rocGNB)/len(auc_rocGNB))


#compute Average performance measure For SVM
print "\n\tAverage  performance measure For SVM"
print 
print "\tAverage Accuracy: = "+ str(sum(accuracy_SVM)/len(accuracy_SVM))
print "\tAverage F1 Score: ="+ str (sum(f1_SVM)/len(f1_SVM))
print "\tAverage AUC ROC:  =  "+ str ( sum(auc_roc_SVM)/len(auc_roc_SVM))

#compute Average performance measure For RF
print "\n\tAverage for  performance measure For RandomForest"
print "\tAccuracy => "+ str( sum(accuracy_RF)/len(accuracy_RF))
print "\tF1 Score => "+ str ( sum(f1_RF)/len(f1_RF))
print "\tAverage AUC ROC:  =  "+ str ( sum(auc_roc_RF)/len(auc_roc_RF))