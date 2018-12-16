#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB();
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


#########################################################


