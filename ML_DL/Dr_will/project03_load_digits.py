# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
# -

print(type(dir(load_digits())))
load_digits().keys()

digit = load_digits()
digit_data = digit.data
print(digit_data.shape)
digit_data[0]

digit_label = digit.target
print(digit_label.shape)
print(digit_label)
digit_label[:20]

print(digit.feature_names)
print(digit.target_names)

# +
# digit.DESCR ??

# +
import matplotlib.pyplot as plt
# %matplotlib inline

plt.imshow(digit.data[0].reshape(8,8), cmap='gray')
plt.axis('off')
plt.show()
# -

for i in range(15):
    plt.subplot(3,5, i+1)
    plt.imshow(digit.data[i].reshape(8,8), cmap='gray')
    plt.axis('off')
plt.show()

new_label = [1 if i == 1 else 0 for i in digit_label]
new_label[:20]

X_train, X_test, y_train, y_test = train_test_split(digit_data, new_label,
test_size=0.3, random_state=15)

# +
#  DecisionTree
decision_tree = DecisionTreeClassifier(random_state=15)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

confusion_matrix(y_test, y_pred)
# -

print(classification_report(y_test, y_pred))

# +
# Random Forest
random_forest = RandomForestClassifier(random_state=32)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# SVM
svm_model = svm.SVC()

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# Stochastic Gradient Descent Classifier (SGDClassifier)
sgd_model = SGDClassifier()

sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))

# +
ls = LogisticRegression()

ls.fit(X_train, y_train)

y_pred = ls.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# random forest가 가장 잘 나옴..
# target이 불균형한 데이터라서 아직도 위의 모델을 채택해야 하는지는 잘 모르겠음.
