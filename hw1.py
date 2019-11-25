import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.io as sio

mnist = sio.loadmat("D:/python_pro/mldata/mnist-original.mat") #手动下载的mnist的数据集
images = mnist["data"].T
targets = mnist["label"].T
X = images/255
Y = targets
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
Y_train=Y_train.ravel()      #transform into vector (row-order first)
Y_test=Y_test.ravel()

from sklearn import metrics

#logistic regression
from sklearn.linear_model import LogisticRegression

LRmodel = LogisticRegression() #default parameters
LRmodel.fit(X_train, Y_train)  #fit the model according to the given training data
train_accuracy = LRmodel.score(X_train, Y_train) #Returns the mean accuracy on the given test data and labels
Y_pred=(LRmodel.predict(X_test)) #Predict class labels for samples in X
test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
print('logistic regression:')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))  #95.83%
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))	#88.60%

#naive bayes
from sklearn.naive_bayes import BernoulliNB

Bmodel=BernoulliNB()   #default parameters
Bmodel.fit(X_train,Y_train)
train_accuracy = Bmodel.score(X_train,Y_train)
Y_pred=(Bmodel.predict(X_test))
test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
print('naive bayes:')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))	#84.10%
print('Testing accuracy: %0.2f%%' % (test_accuracy*100)) 	#83.60%


#support vector machine
from sklearn.svm import LinearSVC

SVMmodel=LinearSVC() #default parameters
SVMmodel.fit(X_train,Y_train)
train_accuracy=SVMmodel.score(X_train,Y_train)
Y_pred=SVMmodel.predict(X_test)
test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
print('SVM:')
print('Training accuracy: %0.2f%%' % (train_accuracy*100)) 	#98.28%
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))	#86.50%

#SVM with another group of parameters
SVMmodel=LinearSVC(penalty='l2',C=0.025)
SVMmodel.fit(X_train,Y_train)
train_accuracy=SVMmodel.score(X_train,Y_train)
Y_pred=SVMmodel.predict(X_test)
test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
print('SVM:')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))   #94.18%
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))	 #90.30%