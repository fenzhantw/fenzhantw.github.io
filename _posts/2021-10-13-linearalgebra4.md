---
title: "[선형대수] 4-1 벡터공간"
excerpt: "벡터공간, 线性代数, 向量空間"
categories:
    - LinearAlgebra

tag:
    - linear algebra
    - Vector space

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---
## 선형대수
선형대수(Linear algebra)는 선형 방정식을 풀기 위한 방법론이다.  2x+3y =0과 같은 선형 방정식이 있으면 해를 만족시키는 x와 y를 찾아내는 것이다. 

이럴 때,방정식의 관계를 이용해 해를 찾아내는 것이 선형대수를 공부하는 목적이다.

## 벡터공간 R^n의 뜻
R = 실수들의 집합이라고 했을때, 벡터공간 R^n은 다음과 같은 뜻을 가진다.
<img src="https://user-images.githubusercontent.com/81638919/137064174-ff229fc5-aab8-4ae8-bcd0-c9d04b495c8e.png"  width="600" height="400">


우선 실수로 이루어진 n개의 순서쌍들의 집합으로 이루어져 있고, 여기서 x1,x2,xn들은 모두 실수에 있다.
R^n에 원소 x가 있을 때, xi를 x의 i-번째 성분이라고 한다. 행렬처럼 생겨먹은 위 그림은 NX1행렬로 이해할 수도 있으며, 가로로도 쓸 수도 있다.

집합 R^n에서는 다음과 같은 연산 두개를 생각 할 수 있다.
1.스칼라 곱과 2.덧셈이라는 연산이다.
![image](https://user-images.githubusercontent.com/81638919/137065653-2d9c9686-403c-4197-83f4-eb80a0fae79a.png)

위 그림과 같이 벡터 x라는 벡터에다가 실수 c스칼라를 곱한다고 했을 때는, 벡터에 있는 모든 성분에다 c배를 한다고 정의하며, 두 벡터를 더한다고 했을 때, 두 벡터들의 각각의 성분들을 더한 것들을 두 벡터의 덧셈, 두 벡터의 합이라고 한다.
여기서 R^n은 그냥 x1,x2,x3,xn들의 집합이라고 생각할 수 있는데, 위에서 정의한 스칼라 곱과 덧셈을 같이 생각할 때 R^n을 벡터공간이라고 부르며 이때, R^n의 원소를 벡터라고 한다.

## 벡터 공간의 성질
u,v,w를 벡터라고 하며, c,d를 스칼라라고 하면 다음과 같은 성질을 만족한다.
(1) u+v = v+u (교환법칙이 성립), u+(v+w) = (u+v)+w (결합법칙이 성립)
(2) u+0 = 0+u = u, u+(-u) = 0
(3) c(u+v)=cu+cv, (c+d)u =cu+du (분배법칙이 성립) 
(4) c(du) = (cd)u, 1u=u (스칼라곱에 대한 결합법칙이 성립)

넓은 의미로 쓰이는 벡터공간은 임의의 집합 V에 스칼라곱과 덧셈에 있어서 위의 조건을 모두 만족하면 V를 벡터공간이라고 부른다. 
ex, 행렬들의 모임 -> 행렬들은 스칼라 배가 있고, 행렬들 두 개를 더할 수 있으니 위의 조건을 만족하기 때문에 행렬들의 모임도 벡터공간으로 이해할 수 있다.
이 강의에서는 벡터 공간 R^n에 집중해보도록 하자.

ex. 벡터공간 R^4를 생각하자.

<img src="https://user-images.githubusercontent.com/81638919/137066597-5fa38fff-7e34-4652-a9fc-2c582f21d49d.png"  width="700" height="200">


## 기하학적인 해석 

R^2에서 좌표평면 입장에서 벡터는 좌표평면상에서 원점과 그 점으로 주어진 화살표로 이해할 수 있다. 
따라서 벡터는 크기와 방향이 있다고 이해할 수 있고, x,y∈R^2, a,b ∈ R에 대해서 ax+by는 원점과 x를 a배한 벡터, 그리고 y를 b배한 벡터가 이루는 평행사변형의 또 다른 꼭지점이라 이해할 수 있다.

예시를 통해 이해해보자.

x=(1,2)가 R^2에 있을 때, 2x=(2,4)이다.이를 좌표평면으로 생각하면 다음과 같다.

![image](https://user-images.githubusercontent.com/81638919/137066741-1f0a714f-dcf0-4d44-a01c-ba64759b42c5.png)
x를 두 배한 2,4는 처음에 주어진 벡터와 방향은 같고 크기가 두 배가 되었다. 그래서 일반적으로 c*x, x를 c배 한 것은 x 좌표평면 입장에서는 x와 방향이 같고 크기를 c배한 벡터로 이해할 수 있다.

x=(1,2),y(2,1)가 R^2에 있을 때, x+y =(3,3)이다. 이를 좌표평면으로 생각하면 다음과 같다.
R^2에서 x+y는 이 벡터들이 이루는 평행사변형을 생각 했을때, x+y는 이 평행사변형에서 원점,x,y 를 제외한 나머지 꼭짓점의 위치라고 생각할 수 있다.

## 부분공간
벡터공간 R^2의 부분집합 H가 스칼라 곱과 덧셈에 대해서 닫혀있고, 벡터 공간의 성질(위의 1~4)을 모두 만족시킬 때 그때 H를 R^n의 부분 공간이라고 한다.

참고 
"연산에 대해 닫혀있다" 수학 개념을 참고할 수 있는 링크

https://terms.naver.com/entry.nhn?docId=2073659&cid=47324&categoryId=47324

부분공간 H는 부분 집합이 되면서 벡터 공간이 되는 것을 말하는데, 아래의 조건을 이용하면 부분공간임을 쉽게 확인할 수 있다.
![image](https://user-images.githubusercontent.com/81638919/138211138-94e83e4b-90e2-459c-a45e-12eb8afaeefd.png)
H가 공집합이 아니면, H에 있는 원소 x,y와 스칼라 c에 대해서 다음과 같이 x+cy가 다시 H에 들어간다면, 모든 x,y와 c에 대해서 이러한 H가 부분공간이 되는 것이다.

![image](https://user-images.githubusercontent.com/81638919/138211316-f32a765d-099f-4a7d-ac0b-50fd849368eb.png)

(1)을 보면, 이 안에 있는 두 개의 원소를 끄집에내서 하나를 스칼라배해서 다시 더한다고 하더라도, 두 번째 있는 성분들은 항상 0이 되기 때문에 다시 이 집합안에 들어간다.
따라서 부분 공간이 된다고 말할 수 있다.

(2)은 A*x=0이 되는 x들을 다 모아놓은 집합이다. 이 안에 있는 벡터를 꺼내보자, 둘 다 A*x=0의 해이다. 여기 x에다가 y에 상수 배를 해서 더하면 역시나 마찬가지로 A를 곱해서 0이 된다.
따라서 부분공간인 것을 증명할 수 있다.


## 일차결합과 생성 집합

위에서 백터공간 R^n에 스칼라 곱과 덧셈을 같이 생각하면 벡터공간이라고 했다. 그렇다면 집합으로서의 R^n과 벡터공간으로서의 R^n의 차이는 무엇인가?
스칼라 곱과 덧셈을 같이 생각한다는 것이 어떤 의미인지 생각해보자

![image](https://user-images.githubusercontent.com/81638919/137431718-5a0a14f6-fe84-4b8f-bef9-25dac4825bbd.png)

위의 3가지 벡터를 생각해면 가장 마지막의 벡터는 앞에 있는 2개의 벡터와 다르다. 이 마지막에 있는 벡터는 스칼라 곱과 덧셈을 이용해서 앞에 있는 두 벡터들로부터 얻을 수 있는 것 처럼, 일반적인 집합에서 집합에 있는 두 원소를 안다고 해도 다른 원소들을 알 방법이 없지만, 벡터 공간에서는 중요한 두 가지 연산 도구들이 있어서 그것을 이용하면 다른 벡터들로 또 다른 벡터를 만들어 낼 수 있다.

따라서 집합으로서의 R^n과 다르게 벡터공간으로서의 R^n은 스칼라곱과 벡터의 합을 통해서 다른 벡터를 만들어 낼 수 있다. 여기서 만들어 낸다는 것은 구어로써 이야기 하는 것이고, 수학적으로 정의하면 다음과 같다.

![image](https://user-images.githubusercontent.com/81638919/137431904-e7ee630a-e655-4c60-9767-b79adb461ba6.png)

만들어 낸다는 개념을 배웠으니, 다음 같은 정의도 자연스럽게 이해할 수 있다.

![image](https://user-images.githubusercontent.com/81638919/138211716-a2cdb479-548e-43fe-9076-76f669b0ae3a.png)

v1부터 vm깨 벡터 m개가 있다. Span v1 to vm이라고하면, v1 to vm으로 만들어 낼 수 있는 모든 일차결합의 벡터들을 다 모아 놓은 집합이라고 한다. 이 집합은 v1부터 vm으로 생성되었다라고 이야기한다.


