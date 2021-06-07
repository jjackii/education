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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

iris.keys()

# Data 준비 (features)
iris_data = iris.data
print(iris_data.shape)
iris_data[0]

# Data 준비 (targets)
iris_label = iris.target
iris_label

print(iris.target_names)
print(iris.feature_names)
print(iris.filename)

print(iris.DESCR)

# +
# print(pd.__version__)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df.head()
# -

iris_df['label'] = iris.target
iris_df.head()

# +
## Learning

# Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=7) # 20% -> test

print('X_train:', len(X_train), 'X_test:', len(X_test),'\n')

print(X_train.shape, y_train.shape)

X_test.shape, y_test.shape

# +
# Classification (분류)

# ML Model : Decision tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# +
# Training
decision_tree.fit(X_train, y_train)

# Test
y_pred = decision_tree.predict(X_test) # input -> features test => predict
print(y_pred) # predicted data
print(y_test) # original data

# +
# Test -> Accuracy( TP+TN / TP+TN+FP+FN )
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy # 30 * 0.9 = 27개 맞춤

# +
## Random Forest
# Decision Tree를 여러개 모아 놓음
# Decision Tree 모델을 여러개 합쳐 놓음으로써 단점을 극복한 모델
# 이러한 기법을 앙상블(Ensemble) 기법이라고 함
# 단일 모델을 여러 개 사용하는 방법 -> 모델 한 개만 사용할 때의 단점을 집단지성으로 극복하는 개념

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# split
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=25)

#
random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))

# +
# Support Vector Machine (SVM)

# +
from sklearn import svm

svm_model = svm.SVC()
print(svm_model._estimator_type)

# +
# Train
svm_model.fit(X_train, y_train)

# Test
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


# +
# Stochastic Gradient Descent Classifier (SGDClassifier)

# +
from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
print(sgd_model._estimator_type)


# +
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# +
# Logistic Regression

# +
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
print(logistic_model._estimator_type)

# +
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))

# +
## Error Matrix

# +
from sklearn.datasets import load_digits

digits = load_digits()
digits.keys()
# -

digits_data = digits.data
print(digits_data.shape)
digits_data[0]

# +
# # %matplotlib inline

plt.imshow(digits.data[0].reshape(8, 8), cmap='gray')
plt.axis('off')
plt.show()

# -

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.data[i].reshape(8, 8), cmap='gray')
    plt.axis('off')
plt.show()


# +
digits_label = digits.target

print(digits_label.shape)
digits_label[:20]


# +
# 입력된 데이터가 3이라면 3을, 3이 아닌 다른 숫자라면 0을 출력
new_label = [3 if i == 3 else 0 for i in digits_label]

new_label[:20]

# +
X_train, X_test, y_train, y_test = train_test_split(digits_data, new_label, test_size=0.2, random_state=15)

decision_tree = DecisionTreeClassifier(random_state=15)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy


# +
# 정확도는 정답의 분포에 따라 모델의 성능을 잘 평가하지 못하는 척도가 될 수 있음(데이터 불균형 정확도는 정답의 분포에 따라 모델의 성능을 잘 평가하지 못하는 척도가 될 수 있음(데이터 불균형)

fake_pred = [0] * len(y_pred)

accuracy = accuracy_score(y_test, fake_pred)
accuracy

# +
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
# -

confusion_matrix(y_test, fake_pred)

print(classification_report(y_test, y_pred))


print(classification_report(y_test, fake_pred)) # 3은 단 하나도 맞추지 못함

# +
# 모델의 성능은 정확도만으로 평가하면 안됨 
# 특히, label이 불균형하게 분포되어있는 데이터는 더 조심

print(accuracy_score(y_test, y_pred))
accuracy_score(y_test, fake_pred)
