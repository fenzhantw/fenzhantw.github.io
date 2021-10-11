---
title: "[선형대수] 3-2 행렬식과 기본행 연산"
excerpt: "행렬의 연산, 线性代数, 矩陣運算"
categories:
    - LinearAlgebra

tag:
    - linear algebra
    - Matrix operation

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---
## 지난 차수 복습
(1) 행렬식과 기본 행 연산

(정의) 행렬식 det : {n x n 행렬 } -> R은 다음을 만족하는 함수로 다음과 같은 조건을 모두 만족시킨다.
![image](https://user-images.githubusercontent.com/81638919/136737199-b86a0fde-a33f-47ce-a20f-7e0062fcbf17.png)

첫 번째, 항등행렬의 행렬식은 1이다.
두 번째, 두 행을 바꾼 행렬의 행렬식은 본래 있는 행렬식의 -를 곱한 것과 같다.
세 번째, 한 행을 상수 배한 행렬의 행렬식은 본래 있는 행렬식의 상수 배와 같다.
네 번째, 한 행이 두 행벡터의 합으로 되어있는 행렬식은 그 두 행벡터를 각각 떨어트린 행렬의 행렬식의 합과 같다.

(2) 행렬식의 성질
![image](https://user-images.githubusercontent.com/81638919/136737236-94250c0d-cdb8-446d-88ce-0c67def3671f.png)

## 일반적인 행렬의 행렬식에 대해서는 어떻게 계산하는가?

1. 기본 행 연산을 이용

->기본 행 연산은
(1) 두 행을 교환한다.
(2) 한 행에 0이 아닌 상수를 곱한다.
(3) 한 행의 상수배를 다른 행에 더한다.

(기본 행 연산) 두 행을 교환한다.
만약, B = 정사각행렬 A의 두 행을 바꿔서 얻은 행렬이고, E는 ln의 두 행을 바꿔서 얻은 기본행렬이라고 한다면
행렬식의 정의에 의해서

![image](https://user-images.githubusercontent.com/81638919/136740869-7aa8d660-410a-4077-bca9-68e910d23b52.png)

(기본 행 연산) 한 행에 0이 아닌 상수를 곱한다.
만약, B = 정사각행렬 A의 한 행에 k를 곱해서 얻은 행렬, E는 ln의 한 행에 k를 곱해서 얻은 기본행렬이라고 한다면
행렬식의 정의에 의해서

![image](https://user-images.githubusercontent.com/81638919/136740898-f7dbb682-6790-4169-8c3b-bfb19ea5ffee.png)


(기본 행 연산) 한 행의 상수배를 다른 행에 더한다.
만약, B = 정사각행렬 A의 한 행의 k배를 다른 행에 더한 행렬, E는 ln의 한 행에 k배를 다른 행에 더한 기본행렬이라고 한다면
행렬식의 성질에 의하여 

![image](https://user-images.githubusercontent.com/81638919/136740914-ccf78665-e823-496d-8eca-39209ae1029c.png)


정리 A가 정사각행렬이고 E가 기본행렬이면 
(1) det E != 0,
(2) det EA = detE* detA,
(3) 일반적으로 기본행렬을 이용해서 다음을 증명할 수 있다.

![image](https://user-images.githubusercontent.com/81638919/136738664-c7833ff5-e096-4d05-9000-1e69d2d502eb.png)


(4) A가 가역행렬이면 det(A^-1) = det(A)^-1,
(5) 일반적으로 det(A^t) = detA.

A가 가역행렬이라 하면 기본 행 연산을 이용해서 얻은 A의 기약 사다리꼴 행렬은 ln이다.
따라서 적당한 기본행행렬 E1....,En= ln 라는 정의를 떠올려 보자.
그러면 1 = detln은 det(E1)*det(E2)...이 되고 각각의 곱이 1이 되기 위해서는 det A는 0이 아니다.

![image](https://user-images.githubusercontent.com/81638919/136739013-49d36c5f-7900-4529-8d7a-39a588f3dcac.png)


A가 가역행렬이 아니라고 한다면, 기본 행 연산을 여러 번 하여 기약사다리꼴로 바꿀 수 있는데, 이 경우 기약사다리꼴이 항등행렬이 아니다.
정사각행렬에서 항등 행렬이 아닌 기약사다리꼴이라면 반드시 맨 아래 0으로만 이루어져 있는 행이 존재해야하고,
그 행렬의 행렬식은 0이 될 것이다.

행렬식의 성질에 의하여 한 행이 모두 0으로 구성된 행이 있으면 0이기 때문이다.
그러면 0= detln은 det(E1)*det(E2)...이 되고 각각의 곱이 0이 되기 위해서는 det A는 0이다.

(정리)
정사각 행렬 A에 대해서 A가 가역행렬이면 행렬식이 0이 아니다.

## 행렬식의 계산법

A에 기본 행 연산을 해서 간단한 행렬(e.g 삼각행렬)B를 얻었다면 위의 관찰과 detB로 부터 det A를 계산할 수 있다. 즉, B라는 행렬식과 앞선 관찰한 기본 행 연산과의 관계를 통해서 A라는 det(A)의 행렬식을 계산할 수 있게 되는 것이다.

(잘 알려진 사실)

행렬식을 계산하는 여러 가지 방법(e.g.여인수 전개)이 있다.
행렬식을 이용해서 연립일차방정식을 해를 구할 수 있다.(크래머 공식)
행렬식을 이용해서 역행렬을 계산할 수 있다.

## 정리

- 행렬식과 기본 행 연산 사이의 관계를 이해
- detAB = detA * detB
- det(A^-1) = det(A)^-1, det(A^t) = detA
- A가 가역행렬 <-> det A !=0

## 문제풀이
<img src="https://user-images.githubusercontent.com/81638919/136743712-6a0e046b-c60d-41ea-8c0b-8490e9d3814f.png"  width="400" height="600"><img src="https://user-images.githubusercontent.com/81638919/136743753-fa2faf50-ce73-45bc-8474-095607aef2d6.png"  width="400" height="600">



