---
title: "[ML] 결정 트리（Decision Tree,決策樹）"
excerpt: "결정 트리, 분류, 分類演算法"
categories:
    - python
    - machine learning

tag:
    - python
    - machine learning

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---
## Intro
결정 트리(의사결정나무라고도 함)는 매우 쉽고 유연하게 적용될 수 있는 알고리즘이다. 또 데이터의 스케일링이나 정규화의 사전 가공의 영향이 매우 적다. 하지만 너무 많은 분류기준은 자료를 과대적합할 위험성이 있는데,
이러한 결정 트리는 다양한 약한 학습기(예측 성능이 상대적으로 떨어지는 학습 알고리즘)을 조합하는 앙상블 기법에서 자주 사용되므로 자세히 알아볼 필요가 있다.
(GBM,XGBoost,LightGBM 등)

## 결정 트리(Decistion Tree)

결정 트리 알고리즘은 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리(Tree)기반의 분류 규칙을 만들어 준다. 이것은 마치 IF-Else 기반의 규칙과 동일한데,

## 균일도

결정 트리에서 정보의 균일도를 측정하는 방법은 크게 두가지가 있는데, 첫번째는 정보 이득(Information Gain) 두번째는 지니 계수가 있다.
먼저 정보 이득은 엔트로피라는 개념을 기반으로 한다. 열역학에서 쓰는 개념으로 무질서 정도에 대한 측도인데, 엔트로피 지수의 값이 클수록 순수도(Purity)가 낮다고 볼 수 있다.
정보 이득 지수는 1에서 엔트로피 지수를 뺀값이며, 1-엔트로피 지수이다. 결정 트리는 이 정보 이득 지수가 높은 기준으로 분할한다.

지니 계수는 경제학에서 불평등 지수를 나타낼 때 사용하는 계수이다. 0이 가장 평등하고, 1로 갈수록 불평등하다. 예를들어 상위 1%가 전체 국민소득의 99%를 번다라고 하면 1로 가까워 질 것이다.
머신러닝에서는 지니 계수가 낮을 수록 균일도가 높은 것으로 해석되어 계수가 낮은 속성을 기준으로 분할한다.

# 결정 트리 주요 하이퍼 파라미터

max_depth : 트리의 최대 깊이를 규정하는데 사용됨.  디폴트는 None

min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합을 제어하는 데 사용됨

min_samples_leaf : 말단 노드의 최소 샘플의 숫자를 지정하며, 과적합을 제어하는 데 사용됨

# Feature 선택 중요도

사이킷런의 DecisionClassifier 객체는 feature_importances_를 통해 중요한 Feature들을 선택할 수 있게 정보를 제공한다.

```python
import seaborn as sns
import numpy as np
%matplotlib inline

# feature importance 추출 
print("Feature importances:\n{0}".format(np.round(dt_clf.feature_importances_, 3)))

# feature별 importance 매핑
for name, value in zip(iris_data.feature_names , dt_clf.feature_importances_):
    print('{0} : {1:.3f}'.format(name, value))

# feature importance를 column 별로 시각화 하기 
sns.barplot(x=dt_clf.feature_importances_ , y=iris_data.feature_names)
```
```
Feature importances:
[0.025 0.    0.555 0.42 ]
sepal length (cm) : 0.025
sepal width (cm) : 0.000
petal length (cm) : 0.555
petal width (cm) : 0.420
```
![image](https://user-images.githubusercontent.com/81638919/134007086-0f3d1c03-206a-46a7-81b5-6540dd127ac3.png)



