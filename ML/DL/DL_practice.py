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

import numpy as np

# # 최소 제곱법 Least Squares Method
# > 단순선형회귀모형 : 모형 내 설명변수(x)가 1개만 있는 모형<br>
# > 1개의 설명변수만으로 반응변수 Y에 대한 영향을 파악하기 위해 사용

# +
# y = a*x + b
# parameter : a, b
# -

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# mean
mx = np.mean(x)
my = np.mean(y)

# LSM (a)분모 : sum of (x - mean(x)) ** 2
divisor = sum([(i - mx)**2 for i in x])


# LSM (a)분자 : sum of (x[i] - mx) * (y[i] - my)
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


dividend = top(x, mx, y, my)

# +
# 기울기 a
a = dividend / divisor

# y절편
b = my - (mx*a)
# -

print('x 평균', mx)
print('y 평균', my)
print('분모', divisor)
print('분자', dividend)
print('\n')
print('기울기(a)', a)
print('y 절편', b)

# # 평균 제곱 오차 Mean Suqare Error, MSE
# > 여러 개의 입력값을 계산할 때<br>
# > 임의의 선을 긋고, 평가 & 수정 (오차 평가 알고리즘)

# +
# 기울기 a, y 절편
fake_a_b = [3, 76]

data = [[2,81], [4,93], [6,91], [8,97]]

x = [i[0] for i in data] # data[0][1] : 81
y = [i[1] for i in data]


# +
# y = ax + b
def predict(x):
    return fake_a_b[0] * x + fake_a_b[1]

# MSE
def mse(y, y_hat):
    return ((y - y_hat) ** 2).mean()

def mse_val(predict_result, y):
    return mse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))
    
print('MSE 최종값: ' + str(mse_val(predict_result, y)))
# -


