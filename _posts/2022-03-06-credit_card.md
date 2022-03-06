---
title: "[BasicML] Credit Card Fault Detection 실습 01"
excerpt: "Kaggle,Fault Detection,LightGBM"
categories:
    - BasicML

tag:
    - python
    - machine learning

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---
## 
이 문서에 나오는 자료와 데이터는 권철민 저의 머신러닝 완벽가이드와 인프런의 강의를 바탕으로 정리한 문서입니다.


#### Credit Card Fault Detection 실습 01

- https://www.kaggle.com/mlg-ulb/creditcardfraud

카드사가 카드 사기건을 검출 및 방지하는 것은 고객이 구매하지 않는 물건에 대해 비용을 지불하지 않도록 하기 위한 매우 중요한 작업임. 따라서 이러한 Fault를 감지하기 위한 ML 모델을 실습을 통해 구축해보고, 성능 테스트를 진행해봄

##### 데이터 호출

```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

card_df = pd.read_csv('./creditcard.csv')
card_df.head(3)
```

| Time |   V1 |        V2 |        V3 |       V4 |       V5 |        V6 |        V7 |        V8 |       V9 |       ... |  V21 |       V22 |       V23 |       V24 |       V25 |       V26 |       V27 |       V28 |    Amount |  Class |      |
| ---: | ---: | --------: | --------: | -------: | -------: | --------: | --------: | --------: | -------: | --------: | ---: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | -----: | ---- |
|    0 |  0.0 | -1.359807 | -0.072781 | 2.536347 | 1.378155 | -0.338321 |  0.462388 |  0.239599 | 0.098698 |  0.363787 |  ... | -0.018307 |  0.277838 | -0.110474 |  0.066928 |  0.128539 | -0.189115 |  0.133558 | -0.021053 | 149.62 | 0    |
|    1 |  0.0 |  1.191857 |  0.266151 | 0.166480 | 0.448154 |  0.060018 | -0.082361 | -0.078803 | 0.085102 | -0.255425 |  ... | -0.225775 | -0.638672 |  0.101288 | -0.339846 |  0.167170 |  0.125895 | -0.008983 |  0.014724 |   2.69 | 0    |
|    2 |  1.0 | -1.358354 | -1.340163 | 1.773209 | 0.379780 | -0.503198 |  1.800499 |  0.791461 | 0.247676 | -1.514654 |  ... |  0.247998 |  0.771679 |  0.909412 | -0.689281 | -0.327642 | -0.139097 | -0.055353 | -0.059752 | 378.66 | 0    |



데이터를 확인해보면, V1,V2...V28까지 있는데, 이는 PCA를 통해 얻는 결과이며 Amount는 카드 사용액이다. Class에 0은 정상, 1은 비정상 사용액이다.

Kaggle의 설명을 보면, 개인정보 때문에 개인을 특정할 수 있는 신용정보 데이터를 가공하여 제공한다고 나와있다.

##### 데이터 가공

```python
from sklearn.model_selection import train_test_split

# 인자로 입력받은 DataFrame을 복사 한 뒤 Time 컬럼만 삭제하고 복사된 DataFrame 반환
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy

# 사전 데이터 가공 후 학습과 테스트 데이터 세트를 반환하는 함수.
def get_train_test_dataset(df=None):
    # 인자로 입력된 DataFrame의 사전 데이터 가공이 완료된 복사 DataFrame 반환
    df_copy = get_preprocessed_df(df)
    
    # DataFrame의 맨 마지막 컬럼이 레이블, 나머지는 피처들
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    
    # train_test_split( )으로 학습과 테스트 데이터 분할. stratify=y_target으로 Stratified 기반 분할
    X_train, X_test, y_train, y_test = \
    train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    
    # 학습과 테스트 데이터 세트 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
```

train_test_split API에서 stratify를 활용하면, 타겟값의 분포도에 따라서 학습과 테스트 데이터를 맞춰서 분할해줌. 아래에서 확인해보자.

```
print('학습 데이터 레이블 값 비율')
print(y_train.value_counts()/y_train.shape[0] * 100)
print('테스트 데이터 레이블 값 비율')
print(y_test.value_counts()/y_test.shape[0] * 100)
```

```
학습 데이터 레이블 값 비율
0    99.827451
1     0.172549
Name: Class, dtype: float64
테스트 데이터 레이블 값 비율
0    99.826785
1     0.173215
Name: Class, dtype: float64
```

##### 평가지표 함수 

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# 수정된 get_clf_eval() 함수 
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```

##### 모델 및 평가지표 확인(로지스틱 회귀)

```python
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)

lr_pred = lr_clf.predict(X_test)
lr_pred_proba = lr_clf.predict_proba(X_test)[:, 1]

# 3장에서 사용한 get_clf_eval() 함수를 이용하여 평가 수행. 
get_clf_eval(y_test, lr_pred, lr_pred_proba)

```

```
오차 행렬
[[85279    16]
 [   60    88]]
정확도: 0.9991, 정밀도: 0.8462, 재현율: 0.5946,    F1: 0.6984, AUC:0.9601
```

imbalnced 된 데이터이기 때문에 모델의 재현율을 중요하게 생각하면 되는데, 재현율이 조금 낮게 나오는 것을 확인할 수 있음

위에서 만들어 놓은 평가지표 함수와 모델을 불러오는 작업을 하나의 함수(get_model_train_eval)로 합치자.

```python
# 인자로 사이킷런의 Estimator객체와, 학습/테스트 데이터 세트를 입력 받아서 학습/예측/평가 수행.
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1]
    get_clf_eval(tgt_test, pred, pred_proba)
    
```

#####  모델 및 평가지표 확인 (lightGBM)

```python
from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

```

```
오차 행렬
[[85290     5]
 [   36   112]]
정확도: 0.9995, 정밀도: 0.9573, 재현율: 0.7568,    F1: 0.8453, AUC:0.9790
```

재현율이 0.75로 로지스틱 모델 보다는 높은 성능이 나왔음.  이제 Feature engenieer를 진행한 후, 다시 모델을 학습하고 평가지표를 확인하자.

##### Amount 피처 가공

```
import seaborn as sns

plt.figure(figsize=(8, 4))
plt.xticks(range(0, 30000, 1000), rotation=60)
sns.distplot(card_df['Amount'])
```

![image](https://user-images.githubusercontent.com/81638919/156914178-0413931d-889f-4a54-a67b-05e0bc860be1.png)


대부분의 금액이 500유로 미만이지만, 나머지 큰 금액이 적지만 존재함. 따라서 긴 꼬리를 그리는 longtail 구조를 가지고 있음. StandardScaler를 이용하여 피처를 변환해보자.

```python
from sklearn.preprocessing import StandardScaler

# 사이킷런의 StandardScaler를 이용하여 정규분포 형태로 Amount 피처값 변환하는 로직으로 수정. 
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))
    
    # 변환된 Amount를 Amount_Scaled로 피처명 변경후 DataFrame맨 앞 컬럼으로 입력
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    
    # 기존 Time, Amount 피처 삭제
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    return df_copy
```

##### StandardScaler 진행 후, 모델 성능 확인

```python
# Amount를 정규분포 형태로 변환 후 로지스틱 회귀 및 LightGBM 수행. 
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('### 로지스틱 회귀 예측 성능 ###')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

```

```
### 로지스틱 회귀 예측 성능 ###
오차 행렬
[[85281    14]
 [   58    90]]
정확도: 0.9992, 정밀도: 0.8654, 재현율: 0.6081,    F1: 0.7143, AUC:0.9702
### LightGBM 예측 성능 ###
오차 행렬
[[85290     5]
 [   37   111]]
정확도: 0.9995, 정밀도: 0.9569, 재현율: 0.7500,    F1: 0.8409, AUC:0.9779
```

앞선 결과와 크게 변한게 없다. 그 다음 Log 변환을 해보자.

##### Log 변환 및 성능 확인

```python
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    # 넘파이의 log1p( )를 이용하여 Amount를 로그 변환 
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    return df_copy
```

```python
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('### 로지스틱 회귀 예측 성능 ###')
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

```

```
### 로지스틱 회귀 예측 성능 ###
오차 행렬
[[85283    12]
 [   59    89]]
정확도: 0.9992, 정밀도: 0.8812, 재현율: 0.6014,    F1: 0.7149, AUC:0.9727
### LightGBM 예측 성능 ###
오차 행렬
[[85290     5]
 [   35   113]]
정확도: 0.9995, 정밀도: 0.9576, 재현율: 0.7635,    F1: 0.8496, AUC:0.9796
```

LightGBM의 재현율이 0.01 정도 높아진것을 확인할 수 있음. Amount Skew되어 있으며, 로그변환을 해서 수행성능이 좋아진 것을 확인할 수 있음.

