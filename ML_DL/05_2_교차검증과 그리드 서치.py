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

# + [markdown] id="C4-mVpQ33b1e"
# # 교차 검증 (Cross validation) 과 그리드 서치 (Grid search)

# + [markdown] id="dVNF7yZjyvoO"
# ## 검증 세트

# + id="banlvMA6RfnM"
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine-date')
# -

wine.info()
wine.head()
wine["class"]

# + id="abR6QA7qRoKl"
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
# -

# ### Training 과 Validation dataset으로 분할

# + id="E-yV4cCXRqNK"
from sklearn.model_selection import train_test_split

train_input, test_input, train_target,test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 1976, "status": "ok", "timestamp": 1591523184001, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="k29hKbw4R7Ki" outputId="ab8158dc-ce49-4c1c-aa0d-d93e185592c5"
print(train_input.shape, test_input.shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 50} executionInfo={"elapsed": 2647, "status": "ok", "timestamp": 1591523184682, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="4VQz-UZ2SeLq" outputId="36d4003f-6961-4b62-ffdc-6c09e0635802"
from sklearn.tree import DecisionTreeClassifier

# Controls the randomness of the estimator.
# the algorithm will select ``max_features`` at random at each split

# max_depth : int, default=None The maximum depth of the tree. If None, then nodes are expanded until
#     all leaves are pure or until all leaves contain less than min_samples_split samples.

#  min_impurity_decrease (IG) : float, default=0.0 : A node will be split if this split induces a decrease of the impurity  greater than or equal to this value.

# min_samples_split : int or float, default=2  The minimum number of samples required to split an internal node:

# 'min_samples_leaf' min_samples_leaf : int or float, default=1. The minimum number of samples required to be at a leaf node.
# leaf node도 조정 가능

dt = DecisionTreeClassifier(random_state=42) # =None : defualt가 있음
dt.fit(sub_input, sub_target)

# Overfitting problem
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# + [markdown] id="Z4gRXnK6y2Pt"
# ## 교차 검증

# + colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"elapsed": 2638, "status": "ok", "timestamp": 1591523184682, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="_J3LId-vSmNH" outputId="7dc8c4d5-0e0f-48a5-86bd-9d703e0bfaf2"
# Evaluate metric(s) by cross-validation and also record fit/score times.
# tree 하나 : node의 변화에 매우 예민 (error propagation 오류 전파) ex.첫 노드가 변하면 아래 다 변함
from sklearn.model_selection import cross_validate
# None, to use the default 5-fold cross validation,
scores = cross_validate(dt, train_input, train_target, cv = 10) # cv:crossvalidation / k-fold (10fold)
print(scores['test_score']) # validation score!!!
# print(scores['test_score'].mean())

# repeated cv (cv를 반복함 / 너무 많이하면 모든 train data를 쓰는거기 때문에 overfitting)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 2630, "status": "ok", "timestamp": 1591523184683, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Yp3aagOoTHsO" outputId="29c0fe86-99df-4d00-96ec-e6c0f6815b0c"
import numpy as np

print(np.mean(scores['test_score']))

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 2613, "status": "ok", "timestamp": 1591523184684, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="1BmP_OTT_agM" outputId="01168ae1-3c6e-4a0a-8527-3b241d889e04"
# 분류모델의 경우 target class의 비율이 잘 유지 되도록
# The folds are made by preserving the percentage of samples for each class.

from sklearn.model_selection import StratifiedKFold # 2분 / 층화 샘플링 -> 비율이 한쪽으로 쏠리지 않게 / data? 뽑을 때 비율을 유지 해주는 거 / regression은 X

# whether to shuffle each class's samples before splitting into batches
# Note that the samples within each split will not be shuffled.
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_validate(dt, train_input, train_target, cv=splitter) # RETURN SCORE
print(np.mean(scores['test_score']))

# + [markdown] id="Q21W8RsqDsDV"
# ## 하이퍼파라미터 튜닝

# +
### Hyperparameter 탐색과 교차검증을 동시에 실시 # hyperparameter에 grid를 깐다?

# + id="S8pqss8onjR5"
from sklearn.model_selection import GridSearchCV

# params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
params = {'max_depth': [2, 3, 4, 5, 6]} # grid search할 hyper para- / dict로 넘겨줘야 함
# max_depth=3

# + id="79MymJqxTu0P"
### 25번의 교차검증을 실시
## 각 params, 5번의 cv 실시 # None(default)
## ``-1`` means using all processors

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv = None, n_jobs=-1) # params을 증가시키면서 search / (none)default=5

# + colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"elapsed": 3834, "status": "ok", "timestamp": 1591523185918, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="tKAlTabkU-Lz" outputId="fd799fa9-44f0-4d6d-de1f-fd194fad9d23"
### 최적의 모델을 이용하여 모델을 다시 훈련

gs.fit(train_input, train_target)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 3825, "status": "ok", "timestamp": 1591523185918, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="q6iX3vH-VeEb" outputId="b7190213-d2c6-4d85-fc32-726b85ac74d5"
dt = gs.best_estimator_
print(dt.score(train_input, train_target))

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 3819, "status": "ok", "timestamp": 1591523185919, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="lIzod3BwVHq-" outputId="f36512fd-b642-43fc-ad8c-5b83604c0874"
print(gs.best_params_)

# + [markdown] id="Hf8ZHWegWf7m"
# ### 5번의 교차검증 결과 확인

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 3812, "status": "ok", "timestamp": 1591523185920, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="0xfQswiui4Tr" outputId="871142d1-f6e6-4380-953a-1273966d9214"
print(gs.cv_results_['mean_test_score'])

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 3804, "status": "ok", "timestamp": 1591523185920, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Rwg2aSyEVO17" outputId="8a4b8fd0-a3b3-47a3-c6a8-1ac726e741fd"
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(best_index)

print(gs.cv_results_['params'][best_index])
# print(gs.cv_results_['params'][0])


# + id="8jHxZ7XmVU11"
import numpy as np

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), # data type(range랑)은 똑같이 array로 나옴 # -1의 차이?
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          } # greed 깔린다. 3차원

# + colab={"base_uri": "https://localhost:8080/", "height": 373} executionInfo={"elapsed": 33403, "status": "ok", "timestamp": 1591523215528, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="KnP3GA6MVsVH" outputId="a4eeaaa1-be8a-4f83-c605-be4fa5333d95"
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1) # ATTRIBUTE

gs.fit(train_input, train_target)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 33395, "status": "ok", "timestamp": 1591523215528, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="qi9-O_VGV0Ho" outputId="e3d3da00-bd1e-4b8f-e681-e42e49868aaa"
print(gs.best_params_) # 노드에 데이터가 12개 이하면 나누지 X

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 33388, "status": "ok", "timestamp": 1591523215529, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="ZnJjLATAV2Sq" outputId="e0c66e0a-b0ff-492f-c2c3-264f1e85c946"
print(np.max(gs.cv_results_['mean_test_score']))

# + [markdown] id="d0k9DQTNlaD6"
# ### 랜덤 서치
# #### 매개변수로서 확률분포객체를 전달

# +
# 양이 너무 많을 경우(시간이 오래걸릴 경우) random하게 뽑아서 grid search -> 대충 그 근방에서 정해진다 함
# cost가 너무 크면 전체적(hybrid하게?)으로 random하게 하고 적당히 min이 잡히면 거기에 맞춰서 grid를 깔고 반복하는 방식

# + id="_T9KTEk1GBcY"
from scipy.stats import uniform, randint

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 33380, "status": "ok", "timestamp": 1591523215530, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="fd0UJpCGGDhz" outputId="a2f0b471-aa50-45f3-d070-76eba4aef24f"
# A uniform discrete random variable

# randint, A uniform discrete random variable.
rgen = randint(low = 0, high = 9)
# rgen.rvs(size = 2)

# + colab={"base_uri": "https://localhost:8080/", "height": 50} executionInfo={"elapsed": 33373, "status": "ok", "timestamp": 1591523215530, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="ch3zTUohIJR6" outputId="c09ba926-927d-4ae1-9e35-433c11ad5014"
# Find the unique elements of an array.
np.unique(rgen.rvs(1000), return_counts=True)
# np.unique(rgen.rvs(1000))

# + colab={"base_uri": "https://localhost:8080/", "height": 50} executionInfo={"elapsed": 33366, "status": "ok", "timestamp": 1591523215531, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="bGhshTn0IjkI" outputId="36e8962b-f517-4521-d272-d829ced45377"
# A uniform continuous random variable.
ugen = uniform(0, 1)
ugen.rvs(10)

# + id="irDX9e6WYTIH"
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }


# params = {'min_impurity_decrease': uniform(0.0001, 0.001),
#           'max_depth': range(5, 20, 1),
#           'min_samples_split': randint(2, 25),
#           'min_samples_leaf': randint(1, 25),
#           }

# + colab={"base_uri": "https://localhost:8080/", "height": 373} executionInfo={"elapsed": 35912, "status": "ok", "timestamp": 1591523218086, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Wc4OIingWQCK" outputId="2356814b-81f8-4edc-e3d1-fb5ea47c0a80"
# In contrast to GridSearchCV, not all parameter values are tried out, but
# rather a fixed number of parameter settings is sampled from the specified
# distributions

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

# + colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"elapsed": 35906, "status": "ok", "timestamp": 1591523218088, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="p7IbsGH3ZSv-" outputId="f922b219-eef0-4c97-bc93-d46b4c6a536e"
print(gs.best_params_)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 35899, "status": "ok", "timestamp": 1591523218089, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="dYI3HwMQbtnr" outputId="2b0f60b9-6fe3-455e-d44a-05c9e892992b"
print(np.max(gs.cv_results_['mean_test_score']))

# + colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 35892, "status": "ok", "timestamp": 1591523218089, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="3QV7yRpidByf" outputId="0db866cd-995e-4d20-9933-7b2c4f249592"
dt = gs.best_estimator_

print(dt.score(test_input, test_target))

# + [markdown] id="cA42IsMdhgE7"
# ## 확인문제

# + colab={"base_uri": "https://localhost:8080/", "height": 373} executionInfo={"elapsed": 37299, "status": "ok", "timestamp": 1591523219504, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="8qxg36iThiUm" outputId="41f7f4cc-9c0a-45ac-f41d-5af9ed18b058"
gs = RandomizedSearchCV(DecisionTreeClassifier(splitter='random', random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

# + colab={"base_uri": "https://localhost:8080/", "height": 87} executionInfo={"elapsed": 37292, "status": "ok", "timestamp": 1591523219505, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="CMZ4UE8ihqwg" outputId="2e3dfbec-7187-4e52-c012-33e21c16a4f6"
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))
