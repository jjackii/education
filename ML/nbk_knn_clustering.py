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

# # 01-3 ~ 02-1 마켓과 머신러닝
#
# ## 생선 분류 문제 예제 (kNN clustering)
# ## 머신러닝 패키지 (sklearn) = http://bit.ly/hg-01-3
# ## 자료분할 (Data spliting) = http://bit.ly/hg-02-1
# ## 특성공학 (Feature engineering) = http://bit.ly/hg-02-2
#
#
# ### python 자료타입: https://wikidocs.net/70

# # k-최근접 이웃 알고리즘 (knn)

# ## package import
#

# +
# ## ! pip install pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Rich output 에 대한 표현 방식
# %matplotlib inline
# -

# ## 도미자료와 방어자료 불러오기

# +
# 도미자료
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 방어자료
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


# list 길이
print(type(bream_length))
print("length of bream = ", len(bream_length), "\nlength of smelt = ",len(smelt_length))

# 용약 통계량
print("mean of 도미길이 = ", round(np.mean(bream_length), ndigits = 2))
print("mean of 방어길이 = ", round(np.mean(smelt_length), ndigits = 2))

print("mean of 도미무게 = ", round(np.mean(bream_weight), ndigits = 2))
print("mean of 방어무게 = ", round(np.mean(smelt_weight), ndigits = 2))
# -

# ## 산점도 그리기 (scatter plot)

# +
# 한글폰트 지정 in matplot
plt.rc('font', family='NanumGothic')


plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight, c = "r")
plt.xlabel('length')
plt.ylabel('weight')
plt.title("도미 vs 방어")
plt.show() # plot all open figures

# -

# ## list 자료를 sklearn 입력 형식에 맞게 수정
# #### 1. 1차원 리스트 -> 2차원 리스트
# #### 2. Feature set과 Target set으로 구성

# +

## Add two list 
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

print(np.shape(length))
print(np.shape(weight))

# print("길이 = ", length[:10])
# print("무게 = ", weight[:10])


# +
## zip function을 이용해서 1차원 list 자료를 2차원 list 자료로 변환 

fish_data_0 = [[l, w] for l, w in zip(length, weight)]

# print(fish_data_0)
print("데이터 타입 fish_data_0 =", type(fish_data_0))
print("배열크기 fish_data_0 = ", np.shape(fish_data_0))


#  numpy 를 이용해서 변환: tuple 
fish_data = np.column_stack((length, weight))
# # print(fish_data)?
print("데이터 타입 fish_data =", type(fish_data))
print("배열크기 fish_data = ", np.shape(fish_data))

print(fish_data[:10,])

# +
# target encoding [1, 0]

fish_target_0 = [1]*35 + [0]*14
print(fish_target_0)


## using np.append
one_index = np.ones(35)
zero_index = np.zeros(14)

print(one_index)
print(zero_index)

fish_target = np.append(one_index, zero_index)

print(fish_target)

# np.concatenate: tuple로 전달
# fish_target = np.concatenate((np.ones(35), np.zeros(14))
# -

# ## sklearn install (기계학습 패키지)

# +
# # ! pip install sklearn

# import knn 클래스 import
from sklearn.neighbors import KNeighborsClassifier

# 클래스 객체 만들기/ learning parameter setting
# n_neighbors : int, default=5
kn = KNeighborsClassifier(n_neighbors = 5)

# ML learning
kn.fit(fish_data, fish_target)


# sklearn classification mean accuracy (score=accuracy)
kn.score(fish_data, fish_target)
# -

# ## 자료분할 (Data splitting)

# +
# fish_data, fish_target

print('bream data 배열 = ',np.shape(bream_length))
print('smelt data 배열 = ',np.shape(smelt_length))

print('fish data 배열 = ',np.shape(fish_data))
print('fish target 배열 = ',np.shape(fish_target))

# +
# Leave one out cross validation
# -

# ### 샘플편향 (Sampling bias)

# +
train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]


print(train_input[:10])
# -

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# ### Random shuffling

# +
np.random.seed(42)

# Return evenly spaced values within a given interval.
index = np.arange(49) # 0~48 int32 형 자료 생성
# index = np.arange(0, 49, 2)

print('index = ',index)


#Randomly permute a sequence in place 
np.random.shuffle(index)

print('index after shuffle = ',index)

# -

# ### 셔플링을 이용한 자료분할

# +
train_input = fish_data[index[:35]]
train_target = fish_target[index[:35]]


test_input = fish_data[index[35:]]
test_target = fish_target[index[35:]]

print(train_input)
# -

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1], c = "r")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)


# +
# Predict the class labels for the provided data.

kn.predict(test_input)
print('Predicted = ', kn.predict(test_input))
print('True = ', test_target)
# -

# ### Learning 사용된 자료 확인하기

print('Features = ', kn._fit_X)
print('Target = ', kn._y)

# ### sklearn 으로 훈련세트와 테스트 세트 나누기

# +

from  sklearn.model_selection import train_test_split

# default test_size = 0.25
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)

print('훈련자료 배열크기 = {0}, 테스트자료 배열크기 = {1}'.format(train_input.shape, test_input.shape))

# train_test_split(
#     *arrays,
#     test_size=None,
#     train_size=None,
#     random_state=None,
#     shuffle=True,
#     stratify=None,
# )
# -

# # 특성공학 (Feature engineering)

# +
# from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# +
# from matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = "^")
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()
# -

# ### 주어진 샘플에서 가까운 이웃찾기

# +
# Finds the K-neighbors of a point.
distance, index = kn.kneighbors([[25, 150]])

# print(kn.kneighbors([[25, 150]]))

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker ="+", c = "b", s = 100)
plt.scatter(train_input[index, 0], train_input[index, 1],
            marker = "D", c = "r")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# -

# ### 이웃한 데이터 포인트찾기

print('훈련자료의 이웃자료 = {0}, \n훈련자료의 그룹 = {1}'.format(train_input[index], train_target[index]))
print('훈련자료와의 거리는 = {}'.format(distance))


# ### 자료의 표준화 (Feature scaling)

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[index,0], train_input[index,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# ### z-score : 모든 자료를 평균이 0이고 표준편차가 1인 자료로 변환

# +
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)
print('평균은 ={0}, 표준편차 = {1}'.format(mean, std))

train_scaled = (train_input - mean)/std

# +
### prediction [25, 150]
new = ([25, 250] - mean)/std
print('z-score 점수는 =', new)

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# +
###
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std

kn.score(test_scaled, test_target)
print('Accuracy of test set= ', kn.score(test_scaled, test_target))
print(kn.predict([new]))
# -

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# # 사용예제

# +

Example
--------
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X, y)

print(neigh.predict([[1.1]]))
# [0]
print(neigh.predict_proba([[0.9]]))
# [[0.66666667 0.33333333]]


# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier()
# neigh.fit(X, y)

# neigh.n_neighbors=3
# print(neigh.predict([[1.1]]))
# # [0]
# print(neigh.predict_proba([[0.9]]))
# # [[0.66666667 0.33333333]]
# -

# # knn 예측

# +

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# -

kn.predict([[30, 600]])

print(kn._fit_X)

print(kn._y)

kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)


# # 확인문제

# +
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

for n in range(5, 50):
    # 최근접 이웃 개수 설정
    kn.n_neighbors = n
    # 점수 계산
    score = kn.score(fish_data, fish_target)
    # 100% 정확도에 미치지 못하는 이웃 개수 출력
    if score < 1:
        print(n, score)
        break

# +
## knn 학습 parameter


#### n_neighbors : int, default=5
#### weights : {'uniform', 'distance'} or callable, default='uniform'
#### algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
#### leaf_size : int, default=30
#### metric : str or callable, default='minkowski'

# -


