---
title: "[BasicML] 교차검증 K-Fold, Stratified K-Fold(k 次交叉驗證)"
excerpt: "교차검증 K-Fold, Stratified K-Fold,機械學習,k 次交叉驗證"
categories:
    - BasicML

tag:
    - python
    - machine learning

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---

## Intro

교차검증은 간단하게 말해서, 학습한 데이터로 여러번 검증을 하는 것을 말한다.

머신러닝은 데이터에 의존적일 수 밖에 없어서, 반복적으로 테스트를 진행하는 교차검증은 필수 요소라고 할 수 있다. 

예를 들어, 학습 데이터와 테스트 데이터가 종속성이 강하다고 치자. 그렇다면 데이터를 학습한 후 진행한 테스트 성능이 좋았을때 알고리즘이 좋아서 그런건지, 비슷한 데이터를 학습해서 성능이 좋은 건지알수 없다. 따라서 여러번의 테스트를 통해 성능을 평가하는 것이 중요하다. 더욱 더 간단하게 예를 들면, 수차례의 모의고사(학습 및 검증)를 수능(성능)을 보기 전에 보는 것이다.



## K-fold? Stratified K-fold?

<img src="https://user-images.githubusercontent.com/81638919/132377909-c911f974-5085-46f6-8f39-eca579acaad3.png" width="450" height="300"/>


Fold는 접는 것을 의미하는데,K는 몇번 접을 것이냐? 라는 의미이다. 여기서 K=5를 주면 총 5개의 폴더 세트를 5번 학습한다는 의미이다. 위 그림과 같이 K=5를 주게 된다면 전체의 5분의 4는 학습용, 검증은 5분의 1로 만드는 것이다.

그리고 검증 평가가 5개가 나오면 평균을 할 수도 있으며, 평균과 같이 데이터를 후속처리 후 최종 평가를 하게 된다.

K-fold는 일반 K-fold와 Stratified K 폴드로 나뉘는데, 불균형한(imbalanced) 분포도를 가진 레이블 데이터 집합을 위한 K 폴드 방식이다.

가령, 1만개의 신용 거래 데이터가 있는데, 그중 1%가 신용 사기 건수라고 하자. 그런데 학습 데이터, 검증 데이터 세트로 나눌때, 100건 밖에 안되기 때문에 학습 데이터에 사기 건수가 한 건도 없을 수가 있다. 반면에 어떤 학습 데이터 세트에서는 사기 데이터가 많을 수 있다.

이렇게 된다면, 학습을 할 수 제대로 할 수 없기 때문에 이렇게 불균형한 데이터를 균일한 분포도를 가지도록 검증 데이터를 추출 하는 작업이다.


## 예제
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])
```
```python
붓꽃 데이터 세트 크기: 150
```
전체의 데이터는 150개이기 때문에, 학습용 데이터는 120개, 검증용 데이터는 30개가 될 것이다.

```python

    n_iter = 0
    
# 1. 위에서 불러온 Kfold 객체에 Split 함수를 호출하면 학습용, 검증용 테스트 데이터 셋의 인덱스를 array로 반환해준다.
# 2. kfold.split( )으로 반환된 array에 담긴 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출한다.
# 3. 추출된 데이터를 위에서 정의한 clf 모델에 학습 및 예측 
# 4. 예측한 데이터의 정확도 측정하여 accuray에 할당
# 5. 학습 회차(n_iter),학습 데이터 사이즈(X_train.shape[0]), 검증 데이터의 크기(X_test.shape[0]) 등을 출력
# 6. cv_accuary에 accuracy 데이터 밀어 넣기
# 7. for loop가 끝나면 cv_accuracy에 밀어 넣은 데이터의 평균을 출력

for train_index, test_index  in kfold.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    #정확도 측정 
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    
    cv_accuracy.append(accuracy)
    
# 개별 iteration별 정확도를 합하여 평균 정확도 계산 
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))
```
```python
#1 교차 검증 정확도 :1.0, 학습 데이터 크기: 120, 검증 데이터 크기: 30
#1 검증 세트 인덱스:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]

#2 교차 검증 정확도 :0.9667, 학습 데이터 크기: 120, 검증 데이터 크기: 30
#2 검증 세트 인덱스:[30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59]

#3 교차 검증 정확도 :0.8667, 학습 데이터 크기: 120, 검증 데이터 크기: 30
#3 검증 세트 인덱스:[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89]

#4 교차 검증 정확도 :0.9333, 학습 데이터 크기: 120, 검증 데이터 크기: 30
#4 검증 세트 인덱스:[ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119]

#5 교차 검증 정확도 :0.7333, 학습 데이터 크기: 120, 검증 데이터 크기: 30
#5 검증 세트 인덱스:[120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149]

## 평균 검증 정확도: 0.8848454545454545
```

## Stratified K-fold를 사용하는 이유

```python
kfold = KFold(n_splits=3)
# kfold.split(X)는 폴드 세트를 3번 반복할 때마다 달라지는 학습/테스트 용 데이터 로우 인덱스 번호 반환. 
n_iter =0
for train_index, test_index  in kfold.split(iris_df):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())

```
```python
## 교차 검증: 1
학습 레이블 데이터 분포:
 1    50
2    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    50
Name: label, dtype: int64
## 교차 검증: 2
학습 레이블 데이터 분포:
 0    50
2    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 1    50
Name: label, dtype: int64
## 교차 검증: 3
학습 레이블 데이터 분포:
 0    50
1    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    50
Name: label, dtype: int64
```

교차 검증 1번을 보면 검증 레이블 데이터 분포에 레이블 0 값만 들어가 있다. 이렇다면 데이터를 통해 0번을 학습할 수 없다.

다른 교차 검증도 마찬가지인것으로 보인다.

## Stratified K-fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
 ```
 ```python
  ## 교차 검증: 1
학습 레이블 데이터 분포:
 2    34
0    33
1    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    17
1    17
2    16
Name: label, dtype: int64
## 교차 검증: 2
학습 레이블 데이터 분포:
 1    34
0    33
2    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    17
2    17
1    16
Name: label, dtype: int64
## 교차 검증: 3
학습 레이블 데이터 분포:
 0    34
1    33
2    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 1    17
2    17
0    16
Name: label, dtype: int64
 ```
 교차 검증 첫번째에 보면 일반적인 K-fold 기법과 다르게 학습 레이블의 분포도가 33,34,33 검증 레이블의 분포도가 17 17 16으로 균일하게 분배된것을 확인할 수 있다.
 
 ## cross_val_score로 간편하게 교차 검증
 ```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , cross_validate
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도(accuracy) , 교차 검증 세트는 3개 
scores = cross_val_score(dt_clf , data , label , scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))

교차 검증별 정확도: [0.9804 0.9216 0.9792]
평균 검증 정확도: 0.9604
```
cross_val_score를 수행하면 반환되는 것은 교차 세트별로 반환한 정확도 값이 나오며, 이를 평균 검증 정확도를 내었다. croos_val_score가 훨씬 더 간단해 보인다.

## 정리
K 폴드는 학습한 데이터로 여러번 검증을 하는 것인데, 학습데이터를 다시 학습 데이터와 검증 데이터로 나누어 반복적으로 테스트하는 것을 말한다.

K 폴드는 다시 일반 K-fold와 Stratified K-fold로 나뉘는데, Stratified는 불균형한 분포도를 가진 레이블 데이터 집합을 위한 K-fold이다.
 
