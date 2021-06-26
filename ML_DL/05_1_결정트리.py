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

# # 결정 트리

# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-mldl/blob/master/5-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩에서 실행하기</a>
#   </td>
# </table>

# + [markdown] colab_type="text" id="gdF762MWpLDx"
# ## 로지스틱 회귀로 와인 분류하기

# + colab={} colab_type="code" id="VuuF90PHgcgs"
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine-date')

# + colab={"base_uri": "https://localhost:8080/", "height": 198} colab_type="code" executionInfo={"elapsed": 1063, "status": "ok", "timestamp": 1590026410431, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="dThiku6olKLY" outputId="1a55da3c-bce5-463c-d122-0af0cb710a73"
wine.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 207} colab_type="code" executionInfo={"elapsed": 859, "status": "ok", "timestamp": 1590026410431, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="ao-fa_VTnauv" outputId="d456fd19-18af-4c47-8c47-caee37a84301"
#Print a concise summary of a DataFrame
wine.info()

# + colab={"base_uri": "https://localhost:8080/", "height": 288} colab_type="code" executionInfo={"elapsed": 648, "status": "ok", "timestamp": 1590026410432, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="_lBE4cRZndrn" outputId="90a30bbf-9fdd-4fd5-d1cd-534c722b7446"
# Describe summary statistics.
# 요약 통계량
wine.describe()
# -

# ### DataFrame to array

# + colab={} colab_type="code" id="ORKbGhMGlQRO"
# ML에 들어가려면 array로 바꿔줘야 함 (sklearn에 들어가기 위해?)
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# + colab={} colab_type="code" id="OMCECWknm3x7"
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42) # random_state : shuffle '42'는 재현을 위함. 안 주면 알아서 섞음

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 440, "status": "ok", "timestamp": 1590026412531, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="iUJ7AGovnYrm" outputId="471bc047-1c12-494c-b801-f0584d73d0f5"
print(train_input.shape, test_input.shape)

# + colab={} colab_type="code" id="lDoSN0sEnrVc"
from sklearn.preprocessing import StandardScaler
# scale하면 scale이 변하지 데이터의 순서가 변하지는 않는다.
# !! 잘라내는 data의 순위는 변하지 않으므로 scale을 할 필요가 없다. -> node가 바뀌지 않는다?

ss = StandardScaler()
ss.fit(train_input) # fit : 평균과 표준편차를 구한다

train_scaled = ss.transform(train_input) # fit한 것을 transform한다
test_scaled = ss.transform(test_input) # train에서 구한 mean,s로 transform / target은 0,1이기 때문에 할 필요 X

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" executionInfo={"elapsed": 680, "status": "ok", "timestamp": 1590026415284, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="hNBO3JgCn7p1" outputId="552a4922-60e1-4bc2-bf6e-24c0f53fe551"
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target)) # score : (confusion matrix 에서의) accuracy !!ㅋㅋ : 맞춘거/전체 accuracy는 편향될 수 있?그래서 a만 보고 판단 X / R^은 뭐였지? 
# 비교적 오버피팅 되었다고 말하긴 어려우나 이것만 갖고 판단 X

# + [markdown] colab_type="text" id="Huyjgx02sS1v"
# ### 설명하기 쉬운 모델과 어려운 모델

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 710, "status": "ok", "timestamp": 1590026442787, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Nnekb2vbsVxL" outputId="da4ad942-dff0-4a6d-c0fb-5ba4e5ca4ef3"
print(lr.coef_, lr.intercept_)

# + [markdown] colab_type="text" id="kfL8p3L5_T-B"
# ## 결정 트리

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" executionInfo={"elapsed": 661, "status": "ok", "timestamp": 1590026465116, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="1yO5owNno9BR" outputId="e4e422f2-6cda-464e-9672-dddcbdb53305"
from sklearn.tree import DecisionTreeClassifier

# random_state: Controls the randomness of the estimator. The features are always randomly permuted at each split
dt = DecisionTreeClassifier(random_state=42) # 1.feaure의 estimation, dsta 뽑아낼 때 / 2.한번에 어떻게 뽑아낼 것인가 
dt.fit(train_scaled, train_target) # fit : 노드?만드는거?

# Accuracy
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# print(test_target)
# (끝까지 나눴기 때문에)  overfitting

# + colab={"base_uri": "https://localhost:8080/", "height": 411} colab_type="code" executionInfo={"elapsed": 35104, "status": "ok", "timestamp": 1590026623533, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="ln3bvp_TpBCW" outputId="7b297daa-8b13-4031-b314-d772f743710c"
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# figsize: Width, height in inches.
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 411} colab_type="code" executionInfo={"elapsed": 1096, "status": "ok", "timestamp": 1590026624637, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="9Bmmuxaa-aRu" outputId="c5051b39-9554-45a7-8e81-6aeccc60a58e"
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# plot_tree(dt, max_depth=1, filled=True)

plt.show()
# sugar(-:scale->성능이똑같다면tree에서는안하는게좋음/scale=ML을안정화시키지만해석이어렵다)
# sugar가 gini impurity를 많이 줄여주기 때문에 계속해서 sugar을 기준으로 나눔 (pc내에서는 이미 다른 것에 대해서도 다 계산한 후임)
# sugat <= -0.239

# + [markdown] colab_type="text" id="uw9MwzTmRAuN"
# ### 가지치기

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" executionInfo={"elapsed": 949, "status": "ok", "timestamp": 1590026661850, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="f8U4ER6L97_O" outputId="50a5057f-b5f5-404e-def9-89ace3d3e03d"
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# + colab={"base_uri": "https://localhost:8080/", "height": 782} colab_type="code" executionInfo={"elapsed": 39778, "status": "ok", "timestamp": 1589973306244, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="QBMxpJA3-A7Q" outputId="dce0291b-37c9-402d-a9a7-727f2005c444"
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" executionInfo={"elapsed": 684, "status": "ok", "timestamp": 1590026704623, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="o0wJS34n_KBW" outputId="a2ab9893-eee6-4f46-e9b1-381398275d73"
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# + colab={"base_uri": "https://localhost:8080/", "height": 782} colab_type="code" executionInfo={"elapsed": 40425, "status": "ok", "timestamp": 1589973306905, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="Kt_biWBq_M-p" outputId="09c4c976-2fb3-48bd-c96e-2408b2a8e9a0"
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
# -

# ### 변수의 중요도 (Feature importance)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" executionInfo={"elapsed": 40423, "status": "ok", "timestamp": 1589973306905, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="odS13_8fymhN" outputId="a917e223-8f4c-494f-a575-cf76de0f08b3"
## Return the feature importances (Ginni importance)
print(dt.feature_importances_) # 변수의 중요도 : alchol = {0}, sugar = {1}, pH = {2}
# 하나의 tree로 결정하는건 위험 -> random forest
# -

dt.feature_importances_

# + [markdown] colab_type="text" id="eDAXu9g61MD5"
# ## 확인문제

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" executionInfo={"elapsed": 639, "status": "ok", "timestamp": 1590026734643, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="t_rqSrgS1OZI" outputId="b480d41b-054d-489b-949d-b51d83cd4635"
dt = DecisionTreeClassifier(min_impurity_decrease=0.0005, random_state=42) # impurity_decrease : infomation gain 보다 낮으면 나누지 마 / depth로 나눈게 아님!
# min_impurity_decrease : float, default=0.0
#     A node will be split if this split induces a decrease of the impurity
#     greater than or equal to this value

dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# + colab={"base_uri": "https://localhost:8080/", "height": 782} colab_type="code" executionInfo={"elapsed": 5868, "status": "ok", "timestamp": 1589974476922, "user": {"displayName": "Haesun Park", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhsWlS7sKQL-9fIkg3FmxpTMz_u-KDSs8y__P1ngQ=s64", "userId": "14935388527648823821"}, "user_tz": -540} id="7BIdDPsv2AOA" outputId="7cb96e18-0423-457c-e7c4-af5b6591c0bf"
plt.figure(figsize=(20,15), dpi=300)
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# +
# CART
# Classification And Regression Tree
