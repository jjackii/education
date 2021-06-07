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
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import LogisticRegression

# +
from sklearn.datasets import load_breast_cancer

b_cancer = load_breast_cancer()
print(type(dir(b_cancer)))
# -

b_cancer.keys()

b_cancer.data[0]

print(b_cancer.feature_names)
print()
print(b_cancer.target_names) # 악성(1) / 양성(0)

print(b_cancer.target)

print(b_cancer.DESCR)

b_cancer_df = pd.DataFrame(data=b_cancer.data, columns=b_cancer.feature_names)
b_cancer_df.head()

b_cancer_df['label'] = b_cancer.target
b_cancer_df.head()

# +
b_cancer_data = b_cancer.data
b_cancer_label = b_cancer.target

X_train, X_test, y_train, y_test = train_test_split(b_cancer_data, b_cancer_label, test_size=0.3, random_state=7)

print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))

# +
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# Random Forest
from sklearn.ensemble import RandomForestClassifier


random_forest = RandomForestClassifier(random_state=32)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# SVM
from sklearn import svm

svm_model = svm.SVC()

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred)) # recall이 낮다?

# +
# Stochastic Gradient Descent Classifier (SGDClassifier)
from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()

sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# Logistic Regression 
from sklearn.linear_model import LogisticRegression

ls = LogisticRegression()

ls.fit(X_train, y_train)

y_pred = ls.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# recall을 기준으로 봤을 때 (유방암 데이터 이므로)
# decision tree는 다른 모델에 비해 점수가 낮아 채택X
# random forest는 다른 모델에 비해 0(양성)을 맞춘 점수는 높지만 1(악성)을 맞춘점수가 좀 더 낮아 채택X
# SVM , SGDClassifier 모델은 recall 1(악성)의 점수가 1. 둘 중 뭐가 더 좋은 모델일까?
