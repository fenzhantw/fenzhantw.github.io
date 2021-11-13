---
title: "[선형대수] 벡터의 내적 - 직교집합"
excerpt: "线性代数"
categories:
    - LinearAlgebra

tag:
    - linear algebra

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---
##
이 문서는 서울시립대학교 박의용 교수님의 강의인 기초 선형대수학을 정리한 내용입니다.

## 복습
내적이라는 것은 두 벡터가 있으면, 두 벡터의 각각의 성분의 곱의 합으로 정의된다.
특히나 두 벡터가 직교한다는 것은 두 벡터의 내적이 0이라는 것과 동치라는 것을 말했다. 오늘은 직교집합에 대해서 이야기해보자.

## 직교집합의 정의
![image](https://user-images.githubusercontent.com/81638919/141646135-37c992b7-ebf1-454f-b53e-28b2865db7c9.png)

즉, 내적해서 항상 0이 되는 직교집합은 벡터들이 서로 수직이 되는 벡터들의 집합이라는 이야기이다.
정규 직교집합이라는 뜻은 S가 직교집합이면서 모든 벡터가 단위벡터라고 하면 정규직교집합이라고 한다.

![image](https://user-images.githubusercontent.com/81638919/141648092-f44b7593-1659-45b4-a7d7-c454c4bca32a.png)


위의 예시 처럼 u,v,w 세 백터는 서로 다른 어떤 벡터와도 내적하더라도 0이되니 집합은 위의 정의 처럼 정규 직교집합이된다.
그럼 이러한 정규 직교집합이 있으면 어떤 좋은 점이 있는가?

0이 아닌 벡터들이 직교집합이라고 한다면 일차독립이라는 것을 바로 알 수 있다.

![image](https://user-images.githubusercontent.com/81638919/141647927-6ead113f-4dd1-4109-90fd-db295bf5bb63.png)


일차독립이라는 것은 앞에 적당한 스칼라들을 곱하고 더해서 0이 되었을 때, 그 앞에 있는 스칼라가 모두 0일 수 밖에 없다는 것을 보여줘야 하는데 직교집합일 경우
그런것을 보여줄 필요 없이 항상 일차독립이다.

## 직교기저의 뜻
![image](https://user-images.githubusercontent.com/81638919/141646569-f189dc36-7c17-42b9-8744-be467044b0e3.png)

어떤 벡터공간의 기저는 생성집합이면서 일차독립이 되는 것을 기저라고 했다.

직교기저라는것은 기저이면서 직교집합이 되면 직교기저라고 한다. 그럼 직교기저를 알면 어떤 부분이 좋은가?

다음과 같은 상황을 한 번 생각해보자.
![image](https://user-images.githubusercontent.com/81638919/141646705-eff6900e-7654-451d-b7bb-430eff623d82.png)

벡터공간 H가 있다. 운이 좋게도 B가 H의 직교기저라고 하자. H에 있는 벡터하나를 선택하고 이것을 v라고 했을때, 이 v는 당연히 기저들의 일차결합으로 표현된다.
그 때의 상수 값, 스칼라 값이 내적으로써 다음과 같이 주어진다라는 것이 이 정리의 이야기이다.
따라서 우리가 직교기저를 알고 있으면 벡터공간의 어떤 벡터v를 이 기저의 일차결합으로 표현할 때, 스칼라 값을 내적을 통해 간단하게 계산할 수 있다.

![image](https://user-images.githubusercontent.com/81638919/141647973-5aab1743-f112-4908-9eee-ead291a47544.png)

위의 예시와 같이 v를 기저들의 일차결합으로 굉장히 쉽게 표현할 수 있다.이것이 직교기저를 알면 얻을 수 있는 좋은점 중 하나이다.

직교기저가 우리한테 유용하다는 것을 알았다. 그렇다면 이 직교기저들은 어떻게 구하는가?

![image](https://user-images.githubusercontent.com/81638919/141647452-39facd69-69ea-481f-995a-33797447164c.png)

벡터공간 H의 기저를 B라 두고, 기저의 개수가 두 개이니 벡터공간, 부분공간 H의 차원은 2차원이다. 이때 직교기저를 찾아보자.

이렇게 스칼라를 내적들의 분수로 정의하면 정규직교기저를 찾을 수 있다.
![image](https://user-images.githubusercontent.com/81638919/141647447-588997bf-ca47-4ac7-b9c1-655a5a31442f.png)
![image](https://user-images.githubusercontent.com/81638919/141647990-eb8e57a4-6224-46cb-b123-7947806e5b01.png)


k차원 벡터공간에 대해서도 같은 식으로 이야기할 수 있으며, 이것을 일반적인 기저로부터 직교기저를 얻는 방법인 그람-슈미트 직교화 과정이라고 한다.

![image](https://user-images.githubusercontent.com/81638919/141647778-725f6df1-4061-44f3-9933-ffa2960167c9.png)

위에서 정규직교기저를 찾는 아이디어를 활용하면 {u1,....uk}의 아무 벡터 ui와 uj를 뽑아서 내적했을 때 항상 0이 되는 것을 확인할 수 있다.
따라서 ui부터 uk는 H의 직교기저를 얻을 수 있다.

## 문제풀이

<img src="https://user-images.githubusercontent.com/81638919/141652580-e970827b-92ec-4f56-a20e-f1eff0a6117f.png"  width="400" height="600"><img src="https://user-images.githubusercontent.com/81638919/141652592-92d34884-6598-4744-bf29-c5e754027d95.png"  width="400" height="600">




