---
title: "[BasicML] Credit Card Fault Detection 실습 02"
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


#### Credit Card Fault Detection 실습 02

- https://www.kaggle.com/mlg-ulb/creditcardfraud

카드사가 카드 사기건을 검출 및 방지하는 것은 고객이 구매하지 않는 물건에 대해 비용을 지불하지 않도록 하기 위한 매우 중요한 작업임. 따라서 이러한 Fault를 감지하기 위한 ML 모델을 실습을 통해 구축해보고, 성능 테스트를 진행해봄



** 각 피처들의 상관 관계를 시각화. 결정 레이블인 class 값과 가장 상관도가 높은 피처 추출 **

```python
import seaborn as sns

plt.figure(figsize=(9, 9))
corr = card_df.corr()
sns.heatmap(corr, cmap='RdBu')
```

![image](https://user-images.githubusercontent.com/81638919/156915773-e096e05e-1986-409b-a1eb-8d464ae7b799.png)

Class와 높은 상관관계를 가진 'V14' 컬럼을 가지고 이상치 제거를 해보도록 하자.

여기서 정의한 이상치는 제1사분위수 - 1.5IQR, 제3사분위수 + 1.5IQR로 정의하였고, 이러한 데이터를 찾기 위한 함수를 구현하면 아래와 같음

** Dataframe에서 outlier에 해당하는 데이터를 필터링하기 위한 함수 생성. outlier 레코드의 index를 반환함. **

```python
import numpy as np

def get_outlier(df=None, column=None, weight=1.5):
    # fraud에 해당하는 column 데이터만 추출, 1/4 분위와 3/4 분위 지점을 np.percentile로 구함. 
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    
    # IQR을 구하고, IQR에 1.5를 곱하여 최대값과 최소값 지점 구함. 
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    
    # 최대값 보다 크거나, 최소값 보다 작은 값을 아웃라이어로 설정하고 DataFrame index 반환. 
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    
    return outlier_index
    
```

```python
outlier_index = get_outlier(df=card_df, column='V14', weight=1.5)
print('이상치 데이터 인덱스:', outlier_index)
```

```
이상치 데이터 인덱스: Int64Index([8296, 8615, 9035, 9252], dtype='int64')
```

V12 컬럼의 8296,8615,9035,9252에 해당하는 인덱스가 앞에서 정의한 이상치에 해당함을 확인



#### **로그 변환 후 V14 피처의 이상치 데이터를 삭제한 뒤 모델들을 재 학습/예측/평가**

```python
# get_processed_df( )를 로그 변환 후 V14 피처의 이상치 데이터를 삭제하는 로직으로 변경. 
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    
    # 이상치 데이터 삭제하는 로직 추가
    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('### 로지스틱 회귀 예측 성능 ###')
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

```

```
### 로지스틱 회귀 예측 성능 ###
오차 행렬
[[85281    14]
 [   48    98]]
정확도: 0.9993, 정밀도: 0.8750, 재현율: 0.6712,    F1: 0.7597, AUC:0.9743
### LightGBM 예측 성능 ###
오차 행렬
[[85290     5]
 [   25   121]]
정확도: 0.9996, 정밀도: 0.9603, 재현율: 0.8288,    F1: 0.8897, AUC:0.9780
```

각 모델별로 재현율이 많이 향상된 것을 확인할 수 있다. 다음은 SMOTE를 활용하여 오버 샘플링을 적용하여 모델을 학습하고 평가해보자.



#### SMOTE 오버 샘플링 적용 후 모델 학습/예측/평가

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
```

```
SMOTE 적용 전 학습용 피처/레이블 데이터 세트:  (199362, 29) (199362,)
SMOTE 적용 후 학습용 피처/레이블 데이터 세트:  (398040, 29) (398040,)
SMOTE 적용 후 레이블 값 분포: 
 0    199020
1    199020
Name: Class, dtype: int64
```

이를 기존에 y_train.value_counts()해서 얻은 값과 비교해보면, 큰 차이가 있는걸 확인할 수 있음.

```
0    199020
1       342
Name: Class, dtype: int64
```

위 그림과 같이 1(부정 거래)값이 0.001정도로 분포되어 있었는데, 이를 오버 샘플링 방법을 통해 0 클래스 값과 1 클래스 값을 똑같이 만들어 줌

다시 로지스틱 회귀를 통해 모델을 평가해보자. 

```python
lr_clf = LogisticRegression()
# ftr_train과 tgt_train 인자값이 SMOTE 증식된 X_train_over와 y_train_over로 변경됨에 유의
get_model_train_eval(lr_clf, ftr_train=X_train_over, ftr_test=X_test, tgt_train=y_train_over, tgt_test=y_test)

```

```
오차 행렬
[[82937  2358]
 [   11   135]]
정확도: 0.9723, 정밀도: 0.0542, 재현율: 0.9247,    F1: 0.1023, AUC:0.9737
```

재현율을 높아졌지만, 정밀도가 매우 낮아서 부적한 모델인 것을 확인할 수 있음. 반면 LightGBM은 어떻게 될까?

```python
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train_over, ftr_test=X_test,
                  tgt_train=y_train_over, tgt_test=y_test)
```

```
오차 행렬
[[85283    12]
 [   22   124]]
정확도: 0.9996, 정밀도: 0.9118, 재현율: 0.8493,    F1: 0.8794, AUC:0.9814
```

앞선 모델보다 정밀도는 떨어졌지만, 재현율이 높아진 것을 확인할 수 있음. 따라서 재현율이 상대적으로 중요한 모델이었기에, 정밀도를 어느 정도 희생하고 재현율을 높이는 초기 목적인 달성이 되었다고 할 수 있다. 

각각의 성능을 데이터 가공 단계에 따라 정리해보자.

#### 최종 정리

| 데이터 가공        |               | 정밀도 | 재현율 | ROC-AUC |
| ------------------ | ------------- | ------ | ------ | ------- |
| 데이터 가공 없음   | 로지스틱 회귀 | 0.8462 | 0.5946 | 0.9601  |
| 데이터 가공 없음   | LightGBM      | 0.9573 | 0.7568 | 0.9790  |
| 데이터 로그 변환   | 로지스틱 회귀 | 0.8812 | 0.6014 | 0.9727  |
| 데이터 로그 변환   | LightGBM      | 0.9576 | 0.7635 | 0.9796  |
| 이상치 데이터 제거 | 로지스틱 회귀 | 0.8750 | 0.6712 | 0.9743  |
| 이상치 데이터 제거 | LightGBM      | 0.9603 | 0.8288 | 0.9780  |
| SMOTE 오버 샘플링  | 로지스틱 회귀 | 0.0542 | 0.9247 | 0.9737  |
| SMOTE 오버 샘플링  | LightGBM      | 0.9118 | 0.8493 | 0.9814  |



초기 데이터 가공이 없었을때, 모델의 성능과 비교하였을 때, 데이터 가공의 효과는 크다고 말할 수 있다. 특히 이상치 데이터 제거를 통해 재현율을 높일 수 있었으며, 이를 통해 이상치 제거는 모델의 성능을 좋게 만드는 것의 기초라고 생각이 된다.

다음에는 모델을 쌓아 성능을 높이는 스태킹 모델에 대해서 배워보도록 한다.

