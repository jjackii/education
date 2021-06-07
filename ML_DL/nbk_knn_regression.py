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

# # k-최근접 이웃 회귀 (knn regression)
#

# + [markdown] colab_type="text" id="i5J2cFzCrDWT"
# ## 데이터 준비

# + colab={} colab_type="code" id="fL3wuWxD0cH6"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

# + colab={} colab_type="code" id="np5j0UTtJNI_"
## 농어

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

type(perch_length)
perch_length.shape
# -

# ## 산점도 그리기 (Scatter plot )

# + colab={"base_uri": "https://localhost:8080/", "height": 279} colab_type="code" executionInfo={"elapsed": 1097, "status": "ok", "timestamp": 1587904061347, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="gE78Nuog4Eg4" outputId="e2862f05-14c9-4445-b711-1326aca5c020"
# scatter plot

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# -

# ## 훈련 자료와 테스트 자료로 분할

# + colab={} colab_type="code" id="dqSDbM-K4pkB"
from sklearn.model_selection import train_test_split

train_input_0, test_input_0, train_target_0, test_target_0 = train_test_split(
    perch_length, perch_weight, random_state=42)

print('훈련 feature 자료의 배열 = {0},테스트 feature 자료의 배열 = {1},'.format(train_input_0.shape, test_input_0.shape))
# print(train_input.shape, test_input.sha


# -

# ### sklearn 형식에 맞게 2차원 배열로 변환

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 1666, "status": "ok", "timestamp": 1587904061928, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Og1eucsRwzIs" outputId="fbfbad2f-99ed-43f7-aedb-d0cf67410c96"
train_input = train_input_0.reshape(-1, 1)
test_input = test_input_0.reshape(-1, 1)

print(train_input.shape)
print(test_input.shape)

# + colab={} colab_type="code" id="2z-LC4zrxzWL"
# 아래 코드의 주석을 제거하고 실행하면 에러가 발생합니다


# + [markdown] colab_type="text" id="NtmNJ7OqrKy_"
# ### knn regression 클래스 import 

# + colab={} colab_type="code" id="BcPh-Da44lhx"
from sklearn.neighbors import KNeighborsRegressor

# knn regression class 만들기
knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target_0)


# -

# ### 결정 계수 ($ R^2$)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 1644, "status": "ok", "timestamp": 1587904061931, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="yEv88u6LIokr" outputId="1ff7d16b-0231-45ee-e9c8-86c739f2b38c"
# Return the coefficient of determination :math:`R^2` of the prediction.

knr.score(test_input, test_target_0)
# -

# ### 평가모델 함수 import

# + colab={} colab_type="code" id="R8Uju0xGLX3s"
from sklearn.metrics import mean_absolute_error

# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
# mean_absolute_error(y_true, y_pred)

mae = mean_absolute_error(test_target_0, test_prediction)
print(mae)



# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 1637, "status": "ok", "timestamp": 1587904061931, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="QKEf3y-5KVQx" outputId="c083b9d6-6836-4980-fcb7-0723e0cbcf86"
## scikit learn
## https://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn.metrics import mean_squared_error

# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
# mean_absolute_error(y_true, y_pred)

mse = mean_squared_error(test_target_0, test_prediction)
print(mse)

# + [markdown] colab_type="text" id="pLW8kdDv5asl"
# ## 과대적합 vs 과소적합
# * Overfitting: 훈련 date에서는 성적이 좋은데 테스트 data에서 성적이 나쁜 경우  
# * Underfitting: 훈련 date 보다  테스트 data의 성적이 높은 경우 또는 둘 다  낮은 경우

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 1634, "status": "ok", "timestamp": 1587904061932, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="ZoXIfmiAJaNw" outputId="99289bfd-0735-4f96-874b-e52dda1725c4"
print('훈련자료의 `R^2` = ', knr.score(train_input, train_target_0))
print('테스트자료의 `R^2` = ', knr.score(test_input, test_target_0))

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 1628, "status": "ok", "timestamp": 1587904061932, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Jhu9abILLHjq" outputId="7fa6a8cf-1137-4ec6-e3df-14acfcb6be4d"
# 이웃의 갯수를 3으로 설정합니다
knr.n_neighbors = 3

# 모델을 다시 훈련합니다

knr.fit(train_input, train_target_0)

print('훈련자료의 `R^2` = ', knr.score(train_input, train_target_0))
print('테스트자료의 `R^2` = ', knr.score(test_input, test_target_0))

# +
r2_train = np.zeros(20)
r2_test = np.zeros(20)
neighbors_n = np.zeros(20)
for n in range(1, 21):
    knr.n_neighbors = n
    knr.fit(train_input, train_target_0)
    r2_train[n - 1] = knr.score(train_input, train_target_0)
    r2_test[n - 1] = knr.score(test_input, test_target_0)
    neighbors_n[n - 1] = n
    
print(r2_train)
print(r2_test)
print(neighbors_n)

# -

plt.scatter(neighbors_n,  r2_train, label = 'train')
plt.scatter(neighbors_n,  r2_test, c="r", label = 'test')
plt.xlabel('neighbors number')
plt.ylabel('R^2')
plt.legend()
plt.show()

# + [markdown] colab_type="text" id="z-oQeMvC2NnY"
# ## 확인문제

# + colab={"base_uri": "https://localhost:8080/", "height": 851} colab_type="code" executionInfo={"elapsed": 2364, "status": "ok", "timestamp": 1587904062678, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="ICPoeo9c2RLG" outputId="2f7e7afc-3d3b-46fd-d8ee-83b3f16fbc55"
# k-최근접 이웃 회귀 객체를 만듭니다
knr = KNeighborsRegressor()
# 5에서 45까지 x 좌표를 만듭니다
x = np.arange(5, 45).reshape(-1, 1)

# n = 1, 5, 10일 때 예측 결과를 그래프로 그립니다.
for n in [1, 5, 10]:
    # 모델 훈련
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    # 지정한 범위 x에 대한 예측 구하기 
    prediction = knr.predict(x)
    # 훈련 세트와 예측 결과 그래프 그리기
    plt.scatter(train_input, train_target)
    plt.plot(x, prediction)
    plt.title('n_neighbors = {}'.format(n))    
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
