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
import pandas as pd
import matplotlib.pyplot as plt

list(range(10)) # list

np.array([1, 2, 3, 4, 5]) # array

np.array([1, 1.2, 4, 5])

np.array([[1, 2],
         [3, 4]])

np.array([1, 2, 3, 4], dtype='float') 

arr = np.array([1, 2, 3, 4])
arr.dtype

arr.astype(float) # Python list와 다르게 array는 단일타입으로 구성

np.zeros(10, dtype=int)

np.ones((3, 5), dtype=float)

np.arange(0, 20, 2) # interval

np.linspace(0, 1, 5) # 5등분  #표준정규분포

np.random.random((2, 2)) # [0, 1] 범위에서 균일한 분포를 갖는다

np.random.normal(0, 1, (2, 2)) # 정규분포 (평균, 표준편차, 배열)

np.random.randint(1, 10, (2, 2))

x2 = np.random.randint(10, size=(3, 4))
x2

print(x2.ndim)
print(x2.shape)
print(x2.size)
print(x2.dtype)

x = np.arange(7)
x

x[2]
x[0] = 7
x

x = np.arange(7)
x[1:4]

print(x[1:])
print(x[:4])
print(x[::2])

x = np.arange(8)
x.shape

x2 = x.reshape((2, 4))
x2

x2.shape

# +
x = np.array([0, 1, 2])
y = np.array([3, 4, 5])

np.concatenate([x, y])
# -

np.arange(8).reshape((2, 4))

matrix = np.arange(4).reshape(2, 2)
np.concatenate([matrix, matrix], axis=0)

matrix = np.arange(4).reshape(2, 2)
np.concatenate([matrix, matrix], axis=1)

matrix = np.arange(16).reshape(4, 4)
upper, lower = np.split(matrix, [3], axis=0)
matrix

matrix = np.arange(16).reshape(4, 4)
left, right = np.split(matrix, [3], axis=1)


# +
# loop

def add_five_to_array(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = values[i] + 5
    return output

values = np.random.randint(1, 10, size=5)
add_five_to_array(values)

# +
big_array = np.random.randint(1, 100, size=10000000)

big_array + 5
# -

matrix = np.arange(9).reshape(3, 3)
matrix

matrix + 5

matrix + np.array([1, 2, 3])

np.arange(3).reshape((3,1))

np.arange(3).reshape((3,1)) + np.arange(3)

x = np.arange(8).reshape((2, 4))
x



data = pd.Series([10, 20, 30, 40], index = ['가', 'b', 'c', 'd'], name="Title")
data

print(data['b'], '\n')
print(data, '\n')
print(data.가)

population_dict = {
'korea': 5180,
'japan': 12718,
'china': 141500,
'usa': 32676
}
population = pd.Series(population_dict)

population

gdp_dict = {
'korea': 169320000,
'japan': 516700000,
'china': 1409250000,
'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)
gdp

country = pd.DataFrame({
'population': population,
'gdp': gdp
})
country

print(country['gdp'], '\n')
print(type(country), '\n')
print(type(country['gdp']))

gdp_per_capita = country['gdp'] / country['population']
gdp_per_capita # NaN :  Not a number

country['gdp per capita'] = gdp_per_capita
country

# +
# country.to_csv('./country.csv')

# +
# country = pd.read_csv('./country.csv')
# country
# -

# index or row/column
country.loc['korea']

country.loc['korea':'china', 'population']

country.iloc[0]

country.iloc[1:3, :2]

dataframe = pd.DataFrame(columns=['이름', '나이', '주소'])
dataframe.loc[0] = ['서지민', '26', '서울']
dataframe.loc[1] = {'이름':'서뿌꾸', '나이':'25', '주소':'제주'}
dataframe.loc[1, '이름'] = '유니'
dataframe

df = pd.DataFrame(columns=['이름', '나이', '주소'])
df.loc[0] = ["서지민", '26', '서울']
df.loc[1] = ["서뿌꾸", '9', '제주']
df.loc[2] = {"이름": "서지윤", '나이': '28', '주소':'제주'}
df.loc[3] = {"이름": "이지은", '나이': '28', '주소':'몰'}
df

df.loc[1, "이름"]

df['연락처'] = np.nan
df

df.loc[0, '연락처'] = '01052915887'
df

len(df)

df['이름'] #df.이름

df[['이름', '주소', '나이']]

df.info() #df.info : 변수

df.isnull()

df.notnull()

df.dropna()

df

df['연락처'].fillna('번호 없음')

df

df['연락처'] = df['연락처'].fillna('번호 없음')
df

A = pd.Series([2,4,6], index=[0,1,2])
B = pd.Series([1,3,5], index=[1,2,3])
A + B

A.add(B, fill_value=0)

A = pd.DataFrame(np.random.randint(0, 10, (2,2)), columns = list("AB"))
A

B = pd.DataFrame(np.random.randint(0, 10, (3,3)), columns = list("BAC")) # random 함수, radint 서브모듈
B

A+B

A.add(B, fill_value=0)

data = {
    'a': [i+5 for i in range(3)],
    'b': [i**2 for i in range(3)]
}
data

df = pd.DataFrame(data)
df

print(df.a.sum())
print(df.sum())

df.mean()

df = pd.DataFrame({
    'col1': [2,1,9,8,7,4],
    'col2': ['a','a','b',np.nan,'d','c'],
    'col3': [0,1,29,4,3,2]
})

df.sort_values('col3', ascending=False)

df.sort_values(['col2','col1']) #shift+tab , inplace

# Lecture 03
# 조건으로 검색하기
df = pd.DataFrame(np.random.rand(5,2), columns=['a','b']) # rand(n,m) 표준정규분포 난수 생성
df

df['a']<0.5

df[df['a']<0.5]

df[(df['a']<0.5) & (df['b']>0.3)]

df.query('a<0.5 and b>0.3') # sql

df = pd.DataFrame(np.arange(5), columns=['num'])
df


# +
def square(x):
    return x**2

df['num'].apply(square)
# -

df['square'] = df.num.apply(lambda x : x**2)
df

df = pd.DataFrame(columns=["phone"])
df.loc[0] = "010-1234-1235"
df.loc[1] = "공일공-일이삼사-1235"
df.loc[2] = "010.1234.일이삼오"
df.loc[3] = "공1공-1234.1이3오"
df["preprocess_phone"] = ''

df


def get_preprocess_phone(phone):
    mapping_dict = {
        "공":"0",
        "일":"1",
        "이":"2",
        "삼":"3",
        "사":"4",
        "오":"5",
        "-":"",
        ".":"",
    }
    
    for key, value in mapping_dict.items(): # items() : 두가지 인자를 동시에 바꿈!!
        phone = phone.replace(key, value) #upgrade
    return phone


df['preprocess_phone'] = df["phone"].apply(get_preprocess_phone)
df

df = pd.DataFrame({"key":['a','b','c','a','b','c'],
            'data1':range(6),
            'data2':range(6)})
df

df.groupby('key').mean()

df.groupby(['key','data1']).sum()

df

df.groupby('key').aggregate(['min', np.median, max]) # **중요중요

df.groupby('key').aggregate({'data1':'min', 'data2':np.sum})


def filter_by_mean(x):
    return x['data2'].mean()>3


df.groupby('key').mean()

df

df.groupby('key').filter(filter_by_mean)

df.groupby('key').apply(lambda x : x.max() - x.min())

df = pd.DataFrame(
np.random.randn(4, 2), #normal distribution
index=[['A', 'A', 'B', 'B'], [1, 2, 1, 2]],
columns=['data1', 'data2']
)

df

df = pd.DataFrame(
    np.random.randn(4,4),
    columns = [['a','a','b','b'], ['1','2','1','2']]
)
df

df.a

df['a']['1']

# +
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

plt.plot(x, y)
plt.tick_params(colors='white', which='both')

# +
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

plt.plot(x, y)
plt.title("First plot")
plt.xlabel("x")
plt.ylabel("y")
plt.tick_params(colors='white', which='both')

# +
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("First Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.set_dip(300)
fig.savefig("first_plot.png")
# -

fig, ax = plt.subplots()
x = np.arange(15)
y = x ** 2
ax.plot(
    x, y,
    linestyle=":",
    marker="*",
    color="#524FA1"
)
ax.tick_params(colors='white', which='both')

# +
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, linestyle="-")

ax.plot(x, x+2, linestyle="--")

ax.plot(x, x+4, linestyle="-.")

ax.plot(x, x+6, linestyle=":")
ax.tick_params(colors='white', which='both')
# -

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, color="r")
ax.plot(x, x+2, color="green")
ax.plot(x, x+4, color='0.8')
ax.plot(x, x+6, color="#524FA1")
ax.tick_params(colors='white', which='both')

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, marker=".")
ax.plot(x, x+2, marker="o")
ax.plot(x, x+4, marker='v')
ax.plot(x, x+6, marker="s")
ax.plot(x, x+8, marker="*")
ax.tick_params(colors='white', which='both')

x = np.linspace(0,10,1000)
fig, ax = plt.subplots()
ax.plot(x,np.sin(x))
ax.set_xlim(-2, 12)
ax.set_ylim(-1.5, 1.5)
ax.tick_params(colors='white', which='both')

fig, ax = plt.subplots()
ax.plot(x, x, label='y=x')
ax.plot(x, x**2, label='y=x^2')
ax.set_xlabel("x", color="white")
ax.set_ylabel("y", color="white")
ax.legend(
loc='best', # upper right
shadow=True,
fancybox=True,
borderpad=2)
ax.tick_params(colors='white', which='both')

fig, ax = plt.subplots()
x = np.arange(10)
ax.plot(
x, x**2, "o",
markersize=15,
markerfacecolor='white',
markeredgecolor="blue")
ax.tick_params(colors='white', which='both')

# +
fig, ax = plt.subplots()
x = np.random.randn(50) 
y = np.random.randn(50)
colors = np.random.randint(0, 100, 
50)
sizes = 500 * np.pi * \
np.random.rand(50) ** 2

ax.scatter(
    x, y, c=colors, s=sizes, alpha=0.3)
ax.tick_params(colors='white', which='both')
# -

x = np.arange(10)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x, x*2)
ax.tick_params(colors='white', which='both')

# +
x = np.random.rand(3)
y = np.random.rand(3)
z = np.random.rand(3)
data = [x, y, z]

fog, ax = plt.subplots()
x_ax = np.arange(3)
for i in x_ax:
    ax.bar(x_ax, data[i],
    bottom=np.sum(data[:i], axis=0))
ax.set_xticks(x_ax)
ax.set_xticklabels(["A", "B", "C"])
ax.tick_params(colors='white')
# -

fig, ax = plt.subplots()
data = np.random.randn(1000)
ax.hist(data, bins=50) # 확률밀도함수
ax.tick_params(colors='white')


