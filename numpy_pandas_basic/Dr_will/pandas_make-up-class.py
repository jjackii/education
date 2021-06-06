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
# data를 분석, 가공 -> 시각적으로 표현하는 과정
# 1. 방대한 데이터 - 많은 처리과정과 계산 필요
# 2. 그 범위와 대상을 최대한 좁혀 구체적으로
# 3. 어떤 활동 때문에 chart의 line이 오르내리는지 보여줘야 함
# infographic : information + graphic
# 대시보드 : 상황을 모니터링 하거나, 이해를 돕거나, 둘 다에 사용하는 데이터의 시각적 표시
# Chart : data를 Graphic적으로 표현한 것
# : Comparison and Ranking(비교,순위) / Part to whole(전체 중 어떤 파트에 속해?) 
# / Trend(경향, 추세) / Correlation(상관관계) / Distribution(범위, 분배, 분산)

# +
## 대시보드 설계
# 시각화 및 게시 프로세스 이해
# 데이터 수집(lxml,csv)/정리(pandas,powerBI) -> 보고서 시각화(R,Python,PowerBI,..) 
# -> 반응형 보고서 만들기(PowerBI)-슬라이서, 필터 -> PowerBI service와 PowerBI mobile로 게시

# +
# 데이터 유형별 시각화 : 범주형(명목형Nominal) / 순서형 // 정량적(tncl) 데이터
# 시각화 - 색상 : 순차적 / 발산형 / 범주형 / 강조 / 경고
# -


# +
# Pandas : python programming 을 위해 만들어진 data analysis & 조작 tool (library)
# Series : 1차원 /  DataFrame : 2차원
# -


# +
# qcut() : bin coloumn을 n개의 bucket으로 변화시킴, 숫자형 데이터를 n개의 범주로 구분할 때 사용
# clip() : 임계치 값을 지정해서 값을 변화시켜줄 때
# abs() : 절대값

# +
# pd.melt : 열의 값을 모아서 행으로 변경
# pd.pivot : 행의 값을 열의 값으로 변경
# -

import pandas as pd

# +
# pd.melt?
# -

df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                  'B': {0: 1, 1: 3, 2: 5},
                  'C': {0: 2, 1: 4, 2: 6}})
df

pd.melt(df, id_vars=['A'], value_vars=['B'])

pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])

df3 = pd.melt(df, value_vars=['A', 'B', 'C'])
df3

# Method Chaining
df3.rename(columns={'variable':'var',
                   'value':'val'})
df3

# +
# df.pivot?
# -

df4 = pd.DataFrame({'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                  'baz': [1, 2, 3, 4, 5, 6],
                  'foo': ['one', 'one', 'one', 'two', 'two', 'two']})

df4

df4.pivot(index='foo', columns='bar', values='baz')

df4.pivot(index='foo', columns='bar', values='baz').reset_index()

df5 = df4.pivot(index='foo', columns='bar', values='baz').reset_index()
df5.melt(id_vars=['foo'], value_vars=['A', 'B', 'C'])

df5.melt(id_vars=['foo'], value_vars=['A', 'B', 'C']).sort_values(by=['foo','bar'])



df1 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']],
                  columns=['animal', 'name'])
df1

df_1 = pd.DataFrame([1], index=['a'])
df_1

df_2 = pd.DataFrame([2], index=['a'])
df_2

# +
pd.concat([df_1, df_2], verify_integrity=True)

# Indexes have overlapping values: Index(['a'], dtype='object')
# key값 중복
# -

adf = pd.DataFrame({'x1': ['A', 'B', 'C'], 'x2': [1, 2, 3]})
adf

bdf = pd.DataFrame({'x1': ['A', 'B', 'D'], 'x3': ['T', 'F', 'T']})
bdf

pd.merge(adf, bdf, how='left', on='x1') 
# default : inner join
# left join (ABC로 병합), on은 key 중심으로 병합

pd.merge(adf, bdf, how='right', on='x1') 

pd.merge(adf, bdf, how='inner', on='x1') 

pd.merge(adf, bdf, how='outer', on='x1') 

adf.x1

bdf.x1

adf.x1.isin(bdf.x1)

adf[adf.x1.isin(bdf.x1)]

adf[~adf.x1.isin(bdf.x1)] # ~을 제와하고 데이터 가져옴

pd.merge(adf, bdf, how='outer', indicator=True) 

pd.merge(adf, bdf, how='outer', indicator=True).query('_merge == "left_only"')

pd.merge(adf, bdf, how='outer', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])



# +
# 데이터 집계활용
# -

import seaborn as sns

df = sns.load_dataset('mpg')
df.head(3)

df.groupby(by='origin').size()

df['origin'].value_counts()

df_mpg = df.groupby(by='origin')
df_mpg.min()

df_mpg.mean()

df.groupby(by='origin')['cylinders'].size()

df.groupby(by='origin')['cylinders'].mean()

df.groupby(by='origin')['cylinders'].median()

df.groupby(['model_year', 'origin'])['cylinders'].mean()

df_sh = pd.DataFrame(
[[4, 7, 10],
[5, 11, 8],
[6, 9, 12]],
index=[1, 2, 3],
columns=['a', 'b', 'c'])
df_sh

# 아래 행 방향으로 하나씩 shift(이동)
df_sh.shift(1)

# 위 행 방향으로 하나씩 shift(이동)
df_sh.shift(-1)

# 특정 column을 이동
df_sh['b'].shift(-1)

df_sh['b'].shift(2)

df['model_year']

df.describe()

# rank : 순위
df['model_year'].rank(method='min')

df['model_year'].rank(method='min').value_counts()

df['model_year'].rank(method='max')

df['model_year'].rank(pct=True) # percentage

df['model_year'].rank(method='first').head()

df_sh

# Cumulative Sum 누적합
df_sh.cumsum()

df_sh.cummin()

df_sh.cummax()

# Cumulative Product 누적곱
df_sh.cumprod()


