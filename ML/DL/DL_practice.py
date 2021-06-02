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


