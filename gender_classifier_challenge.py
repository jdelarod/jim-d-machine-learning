# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:41:11 2018

@author: james.delaroderie
"""

from sklearn import tree

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

clf = tree.DecisionTreeClassifier()

clf_svm = svm.SVC()

clf_nb = GaussianNB()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf_svm = clf_svm.fit(X, Y)
clf_nb = clf_nb.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
prediction_svm = clf_svm.predict([[190, 70, 43]])
prediction_nb = clf_nb.predict([[190, 70, 43]])


# CHALLENGE compare their reusults and print the best one!

print(prediction)
print(prediction_svm)
print(prediction_nb)