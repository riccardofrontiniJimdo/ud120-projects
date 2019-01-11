#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/home/riccardofrontini/git/ud120-projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

############################
### Reduced training set

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
############################

# clf = SVC(C=1.0, kernel='linear');

############################
### C parameter optimization for 'rbf' kernel
#for clsz in [10, 100, 1000, 10000]:

############################

for clsz in [10000]:
    print"++ Class Size:  ", clsz, "  ++"
    clf = SVC(kernel='rbf', C=clsz)
    tt = time()
    clf.fit(features_train, labels_train)
    print "Training time:", round(time()-tt, 3), "s"
    
    tp = time()
    pred = clf.predict(features_test)
    print "Prediction time:", round(time()-tp, 3), "s"
    
    ts = time()
    accuracy = clf.score(features_test, labels_test)
    print accuracy
    print "Scoring time:", round(time()-tp, 3), "s"

print(pred[10])
print(pred[26])
print(pred[50])

###### How many emails are predicted as 'Chris'? ( pred[i] == 1)
sum(pred)
