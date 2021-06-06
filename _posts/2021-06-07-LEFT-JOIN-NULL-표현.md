---
title: "[Mysql] LEFT JOIN할 때 COUNT 집계함수 NULL값 또는 0인 값 표시하는 방법 및 쿼리"
excerpt: "SQL"
categories:
    - SQL

tag:
    - SQL
    - Mysql

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---

## Intro
BI 툴로 데이터를 그래프를 표현할때나, 누군가에게 데이터를 보고할때 COUNT로 집계가 되는 데이터가 없지만 이를 ROW로 나타내고 "0"인 값으로 표현해야 하는 경우가 생긴다.


## NULL값이 표현되지 않는 쿼리문
```
SELECT B.id,B.title,COUNT(A.id)
FROM sales A LEFT JOIN items B ON A.item_id = B.id
GROUP BY A.item_id
```
* 상품명과 판매개수를 구하려는 쿼리를 실행하기 위해 간단한 LEFT JOIN으로 테이블을 JOIN하고 쿼리를 날린다.
* 하지만 다음 쿼리 실행시 COUNT로 집계가 되지 않는 상품의 경우 NULL 값이 출력되지 않고, COUNT(A.id) = 1인 값부터 출력이 된다.

## 서브쿼리를 이용한 NULL값 표현

```
SELECT
  B.id,
  B.title,
  IFNULL(A.id,0)
  FROM items as B
  LEFT JOIN (
    SELECT
      COUNT(A.id)
      FROM sales as A
      WHERE 1=1
      AND 조건
      GROUP BY A.item_id
  ) as B on (A.item_id = B.id)
GROUP BY B.id
```
* 다음과 같이 COUNT를 먼저 구한 서브쿼리에 LEFT JOIN을 하면 COUNT(A.id)가 NULL인 값도 출력이 된다.
* NULL값을 IFNULL 함수를 이용하여 0으로 치환하면 구하고자한 판매개수가 0인 상품명 값을 구하는 쿼리를 출력할 수 있다.

