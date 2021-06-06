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

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

iris.head()

type(iris)

iris.info()

iris.describe(include='all')

iris.species.value_counts()

iris.corr() # 1:일치, 0:불일치, -1:반비례

iris.columns

iris[['sepal_length', 'petal_length', 'petal_width']].corr()

sns.get_dataset_names()

# iris.groupby(iris['species']).mean()
df = iris.groupby(iris['species']).mean()

df.plot()

df.plot.bar() # df.plot.bar(rot = 0)

iris.plot.scatter(x = 'sepal_length', y = 'sepal_width')
# x,y축 지정 -> 산점도는 기본적으로 2차원

x = np.where(iris['species'] == 'setosa', 'red', 'green')
# setosa-red, others-green

iris.plot.scatter(x = 'sepal_length', y = 'sepal_width', c = x)

x = np.where(iris['species'] == 'setosa', 'red',
            np.where(iris['species'] == 'versicolor', 'green', 'blue'))

iris.plot.scatter(x = 'sepal_length', y = 'sepal_width', c = x)

iris.plot.scatter(x = 'petal_length', y = 'petal_width', c = x)



tips = sns.load_dataset('tips')
tips.head()

x = np.where(tips['day'] == 'Thur', 'red',
    np.where(tips['day'] == 'Fri', 'orange',
    np.where(tips['day'] == 'Sat', 'green', 'blue')))

tips.plot.scatter(x = 'total_bill', y = 'tip', c = x)

sns.pairplot(tips)

# hue : ''를 기준으로 색깔을 구별
sns.pairplot(tips, hue = 'day', height=3)

sns.barplot(x = 'day', y = 'tip', data = tips)

sns.barplot(x = 'day', y = 'tip', hue = 'smoker', data = tips)

sns.lmplot(x = 'total_bill', y = 'tip', hue = 'smoker', data = tips)

sns.lmplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'size', col_wrap = 3, data = tips)

pivot = tips.pivot_table(index = 'day', columns = 'smoker', values = 'tip')
pivot

sns.heatmap(pivot, annot = True, cmap = 'Blues')

sns.relplot(x = 'total_bill', y = 'tip', hue = 'day', data = tips)

# +
# # ? sns.relplot(x = 'total_bill', y = 'tip', hue = 'day', col = 'size', data = tips)
# -

sns.catplot(x = 'day', y = 'total_bill', data = tips) # category


