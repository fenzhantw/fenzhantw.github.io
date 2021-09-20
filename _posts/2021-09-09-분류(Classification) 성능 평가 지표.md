---
title: "[ML] 분류 성능 평가 지표（分類器評估方法）"
excerpt: "정확도, 오차행렬, 정밀도, 재현율, F1스코어, ROC AUC 分類器評估方法"
categories:
    - ML

tag:
    - python
    - machine learning

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---
## Intro

분류기를 만들면, 분류기의 성능을 측정해야하는데 일반적으로는 실제 결과 데이터와 예측 결과 데이터가 얼마나 정확하고 오류가 적게 발생하는가에 기반하지만 단순히 정확도만 가지고 판단했을 경우에는 함정에 빠질 수 있으니 유의해야한다. 

## 정확도의 함정
![image](https://user-images.githubusercontent.com/81638919/133983596-c3369ac1-84ee-43b8-949c-2512acf1efe1.png)

정확도는 전체 예측한 데이터 건수중에 예측이 정확히 된 데이터 건수를 비율(%)로 나타낸다. 이러한 정확도는 직관적으로 모델 예측 성능을 나타내는 지표이긴 하나, 데이터셋의 구성에 따라 분류기의 성능을 왜곡할 수 있기 때문에 정확도 수치 하나만 가지고 성능을 평가하지 않는다.

특히, 분류의 결정값(Target) 데이터들이 불균형한 데이터 일때, 정확도는 정확하지 않다.  예를 들어, 신용카드 사기 검출에 전체 데이터 세트가 10000건이 있는데, 1%만 사기라고 생각하자. 이러한 경우 복잡한 알고리즘 필요없이 단순히 "모두 정상이라고 예측"을 했을 때, 매우 높은 정확도인 99%가 나올 것이다. 이처럼 정확도 평가 지표는 불균형한 결정값(Target)을 가지고 있는 데이터 세트에서는 성능 수치로 사용해서는 안된다. 따라서 정확도의 한계점을 극복하기 위해 여러 가지 분류 지표와 함께 ML모델을 평가해야 한다.

## 오차 행렬(Confustion Matrix)

오차 행렬은 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표이다.

![image](https://user-images.githubusercontent.com/81638919/133916601-c1f4f9d5-6548-4522-a3d2-9cad49bd3f8a.png)

위 그림에서 
TN는 예측을 Negative(0)로 했으며, 실제도 Negative(0)
FP는 예측을 Positive(1)로 예측을 했지만 실제는 Negative(0)
FN은 예측을 Negative(0)로 했지만 실제는 Positive(1)
TP는 예측을 Positive로 했지만 실제도 Positive라는 의미이다.

이러한 오차 행렬은 sklearn의 metrics 패키지에서 confusion_matrix로 간단히 호출하면 된다.

```python
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 0]

confusion_matrix(y_true, y_pred)
```
```python
array([[2, 0],
       [2, 2]], dtype=int64)
```
이렇게 출력된 오차 행렬은 ndarray형태로 나오며, TN은 array[0,0]로 2, FP는[0,1]로 0, FN은 [1,0]으로 2, TP는 [1,1]로 2에 해당한다. 
풀어서 설명하면, TN은 6개의 데이터 중 Negative 0으로 예측해서 True가 된 결과 2건, FP는 Positive 1로 예측했지만 실제값은 Negative 0인 값이 없으므로 0건 FN은 Negative 0으로 예측했지만 1인 2건, TP는 Positive로 예측했지만 1인값 2건이다. 그리고 일반적으로 불균형한 레이블 클래스를 가지는 이진 분류 모델에서는 중점적으로 찾아야하는 매우 적은 수의 결과값에 Positivie를 설정해 1값을 부여하고, 그렇지 않은 경우 0값을 부여하는 경우가 많다. 예를들어, 사기 행위 예측 모델에서는 사기 행위가 positive 양성으로 1, 정상 행위가 Negative 음성으로 0 값이 결정 값으로 할당되는 것 처럼 말이다.

이제는 오차 행렬에서 파생되는 다양한 지표를 살펴보자.


## 정밀도(Precision), 재현율(Recall)
![image](https://user-images.githubusercontent.com/81638919/133983656-79cf3286-6f15-4ad0-a6d6-b05c05b639fd.png)![image](https://user-images.githubusercontent.com/81638919/133983693-0d4cdf84-54d4-468c-a63a-77b5f6f6c0ea.png)


위에서 신용카드 사기 검출에서 정확도가 왜 제대로 평가 지표로 활용되지 못하는지 말했는데, 불균형한 데이터 세트에서 정확도보다 더 선호되는 평가 지표인 정밀도(Precision)와 재현율(Recall)에 대해서 알아보자.

정밀도(Precision) (TP/(FP+TP))는 예측을 Positive로 한 대상 중(FP+TP)에 예측과 실제값이 Positive로 일치한 데이터(TP)의 비율을 뜻하며, 그리고 재현율(Recall) TP /(FN+TP)은 실제 값이 Positive인 대상 중(FN+TP)에 예측과 실제값이 Positive로 일치한 데이터(TP)의 비율을 뜻한다. 재현율(Recall)은 민감도(Sensitivity) 또는 TPR(True Positive Rate)라고도 불린다.

두 개념이 조금 헷갈리는데, 정밀도는 ML모델이 얼마나 정밀하게 Positive를 분류했냐를 나타내는 지표이며(Positive로 예측한 값들 중 실제값이 Positive로 일치한 값이기 때문에), 재현율은 실제값이 Positive 중에서 ML모델이 실제로 Positive으로 얼마나 잘 분류를 했는지에 대한 비율이다. 

## 업무에 따른 재현율과 정밀도의 상대적 중요도
아직도 헷갈리는 두 개념을 비즈니스 상황에 대입해 생각해보면 조금 감이 잡히기 시작한다. 

재현율은 실제 Positive 중에서 ML모델이 Positive로 분류한 비율인데, 재현율이 상대적으로 더 중요한 지표의 경우는 실제 Positive 양성인데 데이터 예측을 Negative로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우가 있을 수 있다. 예를 들어, 암 진단 같은 경우 실제로 암인데 음성으로 예측을 해버리면 환자의 생명을 위독하게 만드는 경우가 발생할 수 있다. 또
금융사기 판별을 하는데, 금융사기가 아닌 음성인 데이터라고 판단을 해버리면 회사 입장에서 큰 손해가 발생할 수 있다.

정밀도는 얼마나 정말하게 Positive를 예측했냐를 나타내는 지표이며, 실제 Negative 음성인 데이터 예측을 Positive 양성으로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우 상대적으로 중요하다. 예를 들어, 스팸메일을 분류하는 분류기의 경우 실제 Positive인 스팸 메일을 Negative인 일반 메일로 분류하더라도 사용자가 불편함을 조금 느끼는 정도이지만, 실제 Negative인 일반적인 업무메일을 Positive인 스팸 메일로 분류할 경우에는 메일을 아예 받지 못하게 되어 업무에 차질을 생길 수 있기 때문이다.

다시 한 번 재현율과 정밀도 공식을 살펴보면, 재현율 = TP/(FN+TP), 정밀도 = TP/(FP+TP)인데 재현율과 정밀도 모두 TP를 높이는데 초점을 맞추고 있지만 재현율은 FN을 낮추는 데, 정밀도는 FP를 낮추는데 초점을 맞추고 있다는 것을 알 수 있다. (두 값이 떨어지면 두 비율이 올라기기 때문이다.)

이와 같은 특성 때문에 재현율과 정밀도는 보완적인 지표로 분류 성능을 평가하는데 적용된다. 가장 좋은것은 두 지표 모두 높은 수치를 얻는 것이며, 둘 중 어느 한 지표만 높은 경우는 바람직하지 않다.

## 정밀도/재현율 트레이드 오프

비즈니스의 특성상 정밀도 또는 재현율의 중요도가 달라진다는 것을 배웠다. 그렇다면 둘 중하나가 특별히 강조되어야 할 경우는 어떻게 할까? 바로 분류의 결정 임곗값(Threshold)를 조정해 정밀도 또는 재현율의 수치를 높일 수 있다. 하지만 정밀도와 재현율은 트레이드 오프 관계를 가지고 있기 때문에 둘 중 하나를 올리면 다른 수치가 떨어진다. 이를 정밀도/재현율의 트레이드오프(Trade-off)라고 한다. 타이타닉 Dataset을 이용해 두 지표의 트레이드 오프 관계를 확인해보자.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할. 
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test=train_test_split(X_titanic_df, y_titanic_df, \
                                                  test_size=0.2, random_state=0)
lr_clf = LogisticRegression()

lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)

confusion = confusion_matrix( y_test, pred)
accuracy = accuracy_score(y_test , pred)
precision = precision_score(y_test , pred)
recall = recall_score(y_test , pred)
print('오차 행렬')
print(confusion)
print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy , precision ,recall))
```
```python
오차 행렬
[[92 18]
 [16 53]]
정확도: 0.8101, 정밀도: 0.7465, 재현율: 0.7681
```

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )
```
![image](https://user-images.githubusercontent.com/81638919/133980156-ed81d706-b69e-4293-ab55-c27ffabeebe6.png)

threshold X축이 낮아지면 낮아질수록 ML모델이 양성으로 예측할 확률이 높아지기 때문에 재현율(Recall)이 높아지고 정밀도가 낮아진다.



## 정밀도와 재현율의 함정
정밀도와 재현율의 임곗값 변경은 업무 환경에 맞게 해야지 하나의 수치를 높이기 위해서만 임계값을 조정한다면 정확도와 같이 숫자놀음에 불과해진다.
위 그림에서 볼 수 있듯이, 임곗값을 극단적으로 준다면 정밀도나 재현율을 극단적으로 높일 수 있기 때문이다. 따라서 정밀도와 재현율의 수치가 적절하게 조합돼 분류의 종합적인 성능 평가에 사용될 수 있는 평가 지표가 필요하기 위한 F1 스코어가 나오는데 이를 알아보자.

## F1 Score

F1 스코어(F1 Score)는 정밀도와 재현율을 결합한 지표인데, F1 스코어는 정밀도와 재현율이 어느 한쪽으로 치우치지 않는 수치를 나타낼 때 상대적으로 높은 값을 가진다. 

F1 스코어의 공식은 다음과 같다.

![image](https://user-images.githubusercontent.com/81638919/133980907-a3ee3751-faa6-493a-b36d-f7521e173f86.png)



## ROC 곡선(Receiver Operation Characteristic Curve)과 AUC 
ROC 곡선과 이에 기반한 AUC 스코어는 이진 분류의 예측 성능 측정에서 매우 중요하게 사용되는 지표이다.

ROC 곡선은 FPR(False Positive Rate)가 변할때, TPR(True Positive Rate)가 어떻게 변하는지를 나타내는 곡선이다. 
여기서 TPR은 위에서 말한 재현율(Recall)을 나타내고, 따라서 TPR은 TP/(FN+TP)이며 여기서는 민감도로도 불린다.
FRP은 실제 Negative(음성)을 잘못 예측한 비율을 나타낸다. 따라서 실제는 Negative인데 Positive 또는 Negative로 예측한 것 중 Positive로 잘못 예측한 비율이며, 식으로는 FRP = FP(실제 음성을 양성으로 예측) /(FP+TN - 실제 음성) 이다.

```python
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()
    
roc_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1] )
```

![image](https://user-images.githubusercontent.com/81638919/133982182-a96a2ac5-0871-419f-a277-b6cce1d2b53c.png)

이렇게 ROC 곡선은 FPR과 TPR의 변화 값을 보는 데 이용하며, 분류기의 성능을 보는 것은 ROC 곡선 면적에 해당하는 AUC값으로 결정한다. 낮은 FPR에서 높은 TPR을 얻을 수 있느냐가 관건이며, 곡선이 상단의 모서리에 가까이 가면 직사각형이되어 면적이 1이된다. 면적이 1에 가까울수록 좋은 분류성능을 보여준다고 할 수 있다.

```python

pred_proba = lr_clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_proba)
print('ROC AUC 값: {0:.4f}'.format(roc_score))
```
```python
ROC AUC 값: 0.8706
```

