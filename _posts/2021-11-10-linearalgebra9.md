---
title: "[선형대수] 벡터의 내적"
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
##
이 문서는 서울시립대학교 박의용 교수님의 강의인 기초 선형대수학을 정리한 내용입니다.

## R^2의 내적의 뜻과 성질

![image](https://user-images.githubusercontent.com/81638919/141642159-9a537243-4d7e-4042-aecf-48ff5560d5b5.png)

R^2에서 내적은 각각의 성분의 곱을 합한것과 같다.

(예) u =(2,3), v= (-2,5)에 대해서 내적을 하면, 2*(-2)+3*5 = 11이 된다. 

내적의 성질은 아래 그램과 같다.
![image](https://user-images.githubusercontent.com/81638919/141645668-00a3c1dc-21a1-4331-9e65-95d3aa6cab93.png)

각각의 첫 번째 성분들의 곱을 하고, 두 번째 성분들을 곱을 해서 더하면 된다.

내적으로부터 다음과 같은 개념을 생각할 수 있다.
![image](https://user-images.githubusercontent.com/81638919/141642452-47f86da3-fa63-4322-a116-75fa6fcd7646.png)

u의 길이를, u와 u의 내적에 루트를 씌운 거라고 생각하자.

![image](https://user-images.githubusercontent.com/81638919/141645690-5a956890-1713-43c6-b43f-1aa4850543d7.png)


위의 예제에서 시점을 원점이라고 두는 화살표를 표시하면 가로가 4이고, 세로가 3이다. 
그러면 원점에서 (4,3)까지의 거리는 4의 제곱 더하기 3의 제곱의 루트니, 화살표의 길이가 되며 이는 피타고라스 정리를 통해 확인할 수 있다.

이와 비슷하게 두 벡터 v와 u의 거리도 마찬가지로 이해할 수 있는데, 두 벡터 사이의 거리를  다음과 같이 정의한다.

![image](https://user-images.githubusercontent.com/81638919/141642569-430a904f-4ed8-4425-ba7e-dde39e9b5043.png)
![image](https://user-images.githubusercontent.com/81638919/141645701-3bfe5369-6477-42dd-ab82-952c48ac958a.png)


마지막으로 각도에 대해서 이야기 해보자.

![image](https://user-images.githubusercontent.com/81638919/141642945-878b88ed-9793-4ad8-b86a-72e33c27a3b5.png)
![image](https://user-images.githubusercontent.com/81638919/141645735-8acbc542-31ed-48d1-b27c-de4f77b4c712.png)


다음과 같이 정의할 수 있다. 이 사실로부터 우리가 알 수 있는 것은 우리가 내적을 알고 있으면, 두 벡터 사이의 길이와 거리 그리고 각도에 대해서 이야기할 수 있다는 이야기이다.

이러한 아이디어를 R^2에서가 아니고 벡터공간 R^n에서 이야기해보자.
![image](https://user-images.githubusercontent.com/81638919/141642990-306a08e9-c3c8-47d9-85a8-b48769e3a02e.png)

u와 v의 내적은 각각의 성분의 곱의 합으로 정의되며,
u=(2,3,4,5) v=(-2,5,2,-3)이 R^4에 있을 때, u와 v의 내적은 2*(-2) + 3*5 + 4*2 + 5*(-3)으로 4가 된다.
그리고 R^n에서의 내적도 위에서 정리한 R^2의 내적과 동일한 성질을을 만족한다.

내적을 이용하여 R^n의 벡터들의 길이, 각도, 거리를 정의하자.
![image](https://user-images.githubusercontent.com/81638919/141643453-e694a1ab-e913-45b9-baf4-593df3826d9d.png)

이는 R^2에서의 거리와 길이를 정의한것에 확장이다.
![image](https://user-images.githubusercontent.com/81638919/141645741-9e25bd94-f549-4c28-acb9-947b112c22e4.png)

마지막으로 각도에 대해서 알아보자.
![image](https://user-images.githubusercontent.com/81638919/141643624-f5ed14a8-777c-4db9-a93d-4af02877ab89.png)

*코사인 값은 항상 -1과 1사이에 있어야 하는데, 코시 슈바르츠 부등식을 이용하면 항상 코사인 값이 -1하고 1사이에 있다는것을 보장할 수 있다.

두 벡터 사이의 값이 직각이라면, 직교한다라고 이야기하며 u,v가 직교하면 세타가 90도이며, 코사인 값이 0이 된다.
따라서 두 내적값이 0이다라는 것은 영이 아닌 두 벡터 u,v가 직교한다와 동치이다.

