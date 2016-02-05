## Import classes to be used:
 # - the **GaussianNB**, classification algorithm (model) 
 # - the evaluation scheme **cross_validation**
 # - the artificial dataset
 # - performance evaluation metrics
 

#Cross Validation
from sklearn import cross_validation
from sklearn import datasets

#Classification Algorithms
from sklearn.naive_bayes import GaussianNB

# Performance Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

#import matploit (scientific plotting library)
import matplotlib.pyplot as plot

#import utitlity modules
from utilities import gaussian_naive_bayes
from utilities import average_performance_metric
from utilities import write_to_file

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
accuracy_GNB=[]
f1_GNB=[]
auc_roc_GNB=[]

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

	# Training the model, using the Gaussian Navie Bayes classifier
	labels_predictions = gaussian_naive_bayes(features_train,labels_train,features_test)   

	 # Evaluate the performance of the learned model
	gnb_Accuracy= accuracy_score(labels_test, labels_predictions)
	gnb_f1 = f1_score(labels_test,labels_predictions)
	gnb_auc_roc = roc_auc_score(labels_test,labels_predictions)
	
		
	#Store the result for averaged computation
	accuracy_GNB.append(gnb_Accuracy)
	f1_GNB.append(gnb_f1)
	auc_roc_GNB.append(gnb_auc_roc)

#Record per fold performance results in a file
write_to_file("gnb",accuracy_GNB,f1_GNB,auc_roc_GNB)

#compute Average performance measure For GaussianNaiveBayes
print "\n##Average performance measure For GaussianNaiveBayes##\n"
print "\tAverage Accuracy: = "+ average_performance_metric(accuracy_GNB)
print "\tAverage F1 Score: = "+ average_performance_metric(f1_GNB)
print "\tAverage AUC ROC:  = "+ average_performance_metric(auc_roc_GNB)
print

