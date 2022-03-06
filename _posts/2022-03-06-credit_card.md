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

![img](data:image/png;base64,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)

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

