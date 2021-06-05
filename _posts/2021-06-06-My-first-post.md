---
title:  "SQL 코딩테스트 대비 by 프로그래머스"
excerpt: "프로그래머스 SQL 고득점 kit 모든 문제 답안과 오답노트를 정리합니다. 유형별 문제를 금방 복습하는 것이 목표인 글입니다."
toc: true
toc_sticky: true

categories:
  - SQL
tags:
  - SQL
  - 코딩테스트
last_modified_at: 2020-03-13TO20:00:00+09:00
---

# 목차
- [목차](#목차)
- [프로그래머스 SQL 문제](#프로그래머스-sql-문제)
	- [SELECT](#select)
- [SUM, MAX, MIN](#sum-max-min)
	- [GROUP BY](#group-by)
		- [4번부터 어렵다!](#4번부터-어렵다)
	- [IS NULL](#is-null)
	- [JOIN](#join)
		- [JOIN 은 오랜만이라 복습을 좀 했습니다.](#join-은-오랜만이라-복습을-좀-했습니다)
	- [String, Date](#string-date)
		- [DATE_FORMAT() 관련 참고](#date_format-관련-참고)
	- [빠른 복습](#빠른-복습)

# 프로그래머스 SQL 문제
![image](https://user-images.githubusercontent.com/44190293/76614435-693d1580-6563-11ea-9350-05fcf55e228f.png)
- [https://programmers.co.kr/learn/challenges?tab=sql_practice_kit](https://programmers.co.kr/learn/challenges?tab=sql_practice_kit)
- 예전에 IS NULL 까지는 풀었었는데, 하나도기억이 안나서 초기화하여 다시 진행하였습니다.
- 다음에는 이 글만 보고 복습할 수 있도록 정리하였습니다.
- 기존에 한번 공부하신 분들이라면 하루안에 가능합니다.

## SELECT
1. `SELECT * FROM ANIMAL_INS ORDER BY ANIMAL_ID` <br>
   **ORDER BY 문제**<br>
   ORDER BY column_name (ASC, DESC)
   
2. `SELECT NAME, DATETIME FROM ANIMAL_INS ORDER BY ANIMAL_ID DESC`<br>
	**ORDER BY 문제, DESC**<br>
	정렬옵션 내림차순 DESC 쓰는 문제
3. `SELECT ANIMAL_ID, NAME FROM ANIMAL_INS WHERE INTAKE_CONDITION='Sick'`<br>
	**WHERE절 문제** <br>
	기억할 연산자 BETWEEN a AND b, IN list, LIKE, IS NULL
4. `SELECT ANIMAL_ID, NAME FROM ANIMAL_INS WHERE INTAKE_CONDITION != 'Aged'`<br>
	**WHERE절 문제 : != 연산자**
5. `SELECT ANIMAL_ID, NAME FROM ANIMAL_INS ORDER BY ANIMAL_ID`<br>
	**SELECT문 column 선택 문제**
6. `SELECT ANIMAL_ID, NAME FROM ANIMAL_INS ORDER BY ANIMAL_ID`<br>
	**ORDER BY 문제, 다중 정렬 기준**<br>
	예시) SELECT * FROM table ORDER BY 1 ASC, 3 DESC<br>
	이런식으로 순서대로 정렬기준 적용한다 
7. `SELECT NAME FROM ANIMAL_INS ORDER BY DATETIME LIMIT 1;`<br>
	**LIMIT 명령어 알기**

# SUM, MAX, MIN
1. `SELECT MAX(DATETIME) AS '시간' FROM ANIMAL_INS`<br>
	**MAX() 집계함수 알기**
2. `SELECT MIN(DATETIME) FROM ANIMAL_INS ` <br>
	**MIN() 집계함수 알기**
3. `SELECT COUNT(ANIMAL_ID) FROM ANIMAL_INS`<br>
	**COUNT() 집계함수 알기**
4. `SELECT COUNT(DISTINCT NAME) FROM ANIMAL_INS`<br>
	**SELECT DISTINCT 절, DISTINCT의 위치알기**<br>
	GROUP BY는 범주화이므로 오답<br>
	**오답:**`SELECT COUNT(NAME) FROM ANIMAL_INS GROUP BY NAME`<br>
	이러면 이름별 갯수 나오게 됨.

## GROUP BY
1. `SELECT ANIMAL_TYPE, COUNT(ANIMAL_ID) FROM ANIMAL_INS GROUP BY ANIMAL_TYPE`<br>
	**GROUP BY 알기**
2. `SELECT NAME, COUNT(NAME) FROM ANIMAL_INS GROUP BY NAME HAVING COUNT(NAME) >= 2`<br>
	**HAVING절 알기** (그룹화 이후에 조건 적용, 집계함수 가능)
3. `SELECT HOUR(DATETIME), COUNT(DATETIME) FROM ANIMAL_OUTS WHERE HOUR(DATETIME) BETWEEN 9 AND 19 GROUP BY HOUR(DATETIME) `<br>
	**HOUR()등 날짜-시간 함수 알기**,<br> **WHERE, HAVING 차이 제대로 알기**,<br> **HAVING을 하면 그룹화 이후에 그룹마다에 적용하는 거야**
### 4번부터 어렵다! 
[https://codingmoonkwa.tistory.com/26](https://codingmoonkwa.tistory.com/26)
- MySQL 에서 :=, 즉 대입연산자 사용해서 변수 사용하기h
- @변수이름 ex) @hour
- @hour := @hour + 1 이런식으로 하나씩 늘려나감
- SELECT 안에 SELECT문으로 칼럼 한개 return 하는 select문 넣으면 가능한것 같다
- @hour := @hour + 1은 계속 증가하기만 하므로 while 문처럼 loop로 작동하고,
- where절을 통해 exit 조건을 만들어주는 것 같다

(2번째 풀이)
	- 0~23시 모든 시간에서, **해당 시각에 입양된 기록이 없어도(NULL 이어도) COUNT 0으로 출력하는 것이 관건**
	- UNION으로 0~23 모든 시간을 가진 TABLE을 생성하고, 여기에 LEFT JOIN과 IFNULL을 사용해 문제 해결
	- [https://mentha2.tistory.com/97](https://mentha2.tistory.com/97) [행궁동 데이터과학자]

## IS NULL
1. `SELECT ANIMAL_ID FROM ANIMAL_INS WHERE NAME IS NULL`<br>
	**IS NULL을 아는지. WHERE 절에 사용하는 연산자**
2. `SELECT ANIMAL_ID FROM ANIMAL_INS WHERE NAME IS NOT NULL`<br>
	**IS NOT NULL을 아는지.**
3. **틀렸음** `SELECT ANIMAL_TYPE, IFNULL(NAME, 'No name') as NAME, SEX_UPON_INTAKE FROM ANIMAL_INS`<br>
	**IFNULL(컬럼, 대체값) 을 아는지**

## JOIN

### JOIN 은 오랜만이라 복습을 좀 했습니다.
[https://futurists.tistory.com/17](https://futurists.tistory.com/17)
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- OUTER JOIN
- CROSS JOIN
- SELF JOIN
FROM 절에 추가하여 사용한다. <br>
SELECT * FROM 테이블1 JOIN 테이블2 ON 조인조건 <br>
이런식으로

1. `SELECT B.ANIMAL_ID, B.NAME FROM ANIMAL_INS AS A RIGHT JOIN ANIMAL_OUTS AS B ON A.ANIMAL_ID = B.ANIMAL_ID WHERE A.ANIMAL_ID IS NULL` <br>
	**LEFT, RIGHT JOIN 아는지. 조인 이후에 WHERE 적용되는걸 아는지**<br>
	(**오답**) 비교연산자 == 아니다!!! MYSQL에서는 = 이다!
2. `SELECT A.ANIMAL_ID, A.NAME FROM ANIMAL_INS AS A  JOIN ANIMAL_OUTS AS B ON A.ANIMAL_ID = B.ANIMAL_ID WHERE A.DATETIME > B.DATETIME ORDER BY A.DATETIME`<br>
	**ON 절과 WHERE 절 헷갈리지 말자**
3. `SELECT A.NAME, A.DATETIME FROM ANIMAL_INS AS A LEFT JOIN ANIMAL_OUTS AS B ON A.ANIMAL_ID = B.ANIMAL_ID WHERE B.ANIMAL_ID IS NULL ORDER BY A.DATETIME LIMIT 3`<br>
	**ORDER BY >> LIMIT, 상위 N개 레코드만 뽑아 보기**<br>
	**A에는 있지만 B에는 없는 거 찾기 = JOIN**
4. `SELECT A.ANIMAL_ID, A.ANIMAL_TYPE, A.NAME FROM ANIMAL_INS AS A JOIN ANIMAL_OUTS AS B ON A.ANIMAL_ID = B.ANIMAL_ID WHERE A.SEX_UPON_INTAKE != B.SEX_UPON_OUTCOME ORDER BY A.ANIMAL_ID`<br>
	**문제 잘읽고, != 사용하여 다른 레코드 찾기**

## String, Date
1. `SELECT ANIMAL_ID, NAME, SEX_UPON_INTAKE FROM ANIMAL_INS WHERE NAME IN ('Lucy', 'Ella', 'Pickle', 'Rogan', 'Sabrina', 'Mitty')`<br>
	**where 절에서 <컬럼 IN (후보1, 후보2, 후보3...) 아는지**<br>
	**문자열 ' ' 빼먹지말기 **
2. `SELECT ANIMAL_ID, NAME FROM ANIMAL_INS WHERE NAME LIKE '%EL%' AND  NIMAL_TYPE = 'DOG' ORDER BY NAME`<br>
	**WHERE 절 LIKE 알기. 정규표현식 까지는 아니고 %로 다른문자 올 수 있음을 표시하는듯**
3. `SELECT ANIMAL_ID, NAME, IF((SEX_UPON_INTAKE LIKE 'Neutered%') OR (SEX_UPON_INTAKE LIKE 'Spayed%'), 'O', 'X') AS '중성화' FROM ANIMAL_INS ORDER BY ANIMAL_ID`<br>
	MYSQL의 IF문 형식 = EXCEL과 똑같다.<br>
	**IF(조건, 참리턴값, 거짓리턴값), AND, OR 이것도 다 가능**
4. `SELECT A.ANIMAL_ID, A.NAME FROM ANIMAL_INS AS A JOIN ANIMAL_OUTS AS B ON A.ANIMAL_ID = B.ANIMAL_ID ORDER BY (B.DATETIME - A.DATETIME) DESC LIMIT 2`<br>
	ORDER BY 절 안에 서브쿼리를 해야하나 고민했지만 아니다!<br>
	**ORDER BY 안에도 () 사용해서 계산값을 정렬기준화 할 수 있다**
5. `SELECT ANIMAL_ID, NAME, DATE_FORMAT(DATETIME, '%Y-%m-%d') AS '날짜' FROM ANIMAL_INS ORDER BY ANIMAL_ID`<br>
	**DATE_FORMAT() 함수알기!, 포맷지정자 알기**

### DATE_FORMAT() 관련 참고
-  [https://lightblog.tistory.com/155](https://lightblog.tistory.com/155)
- %y면 2개, %Y면 4개, %M이면 글자, %m이면 숫자, %D, %d도 마찬가지
- 날짜 프린트하기도 편하겠다!


## 빠른 복습 
SELECT문
- ORDER BY
- ORDER BY ASC, DESC
- ORDER BY col1 DESC, col2 ASC
- WHERE
- WHERE col BETWEEN 300 AND 500
- WHERE col IN (300, 500)
- WHERE col LIKE 문자열
- WHERE col IS NULL
- WHERE col IS NOT NULL
- LIMIT 상위레코드 n개
- DISTINCT
- GROUP BY

SELECT DISTINCT
- [https://extbrain.tistory.com/109?category=270532](https://extbrain.tistory.com/109?category=270532)
- SELECT DISTINCT 칼럼 FROM 테이블
- SELECT DISTINCT 칼럼 FROM 테이블 WHERE 조건식
- SELECT COUNT(DISTINCT 칼럼) FROM 테이블

HAVING절
- HAVING은 그룹화(GROUP BY) 이후의 조건
- WHERE은 그룹화 이전에 조건
- WHERE절에서는 집계함수 사용할 수 없다
- HAVING절에서는 집계함수 가지고 조건 비교를 할 때 사용된다
- 

집계함수
- Count()를 제외한 집계함수는 Null값을 무시하며
- SELECT문 혹은 HAVING 절에만 사용할 수 있습니다.
- Count()는 Null 값을 제외하고 계산한다.
- COUNT()
- AVG()
- MAX()
- MIN()
- SUM()

날짜, 시간 함수
- HOUR()
- MINUTE()
- SECOND()
- YEAR()
- WEEK()
- MONTHNAME()
- DAYNAME()
- MONTH()
- TO_DAYS()
- DAYOFYEAR()
- DAYOFMONTH()

그외
- SELECT AS = alias( 영어 뜻 : 별칭)
- MYSQL 주석달기 키워드 '--' 

**SELECT문 실행 순서**
1. FROM
2. WHERE
3. GROUP BY
4. HAVING
5. SELECT
6. ORDER BY

**SELECT문 정의 순서**
7. SELECT
8. FROM
9. WHERE
10. GROUP BY
11. HAVING
12. ORDER BY 
