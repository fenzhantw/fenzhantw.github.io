---
title: "[pandas] Dictionaly와 list를 이용한 데이터프레임 생성 "
excerpt: "pandas, list of dict, dict of list"
categories:
    - python
    - pandas

tag:
    - python
    - machine learning

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---

## intro
pandas를 이용하려면 2차원 배열로 표현되는 데이터프레임 형식으로 바꿔야 한다. 가장 많이 사용되는 딕셔너리와 리스트 형식을 데이터 프레임으로 변경해보자

![image](https://user-images.githubusercontent.com/81638919/132942510-7ad9a61e-e336-4dbb-8f5b-a4a1a33846cf.png)

위 같은 데이터프레임을 만들어본다고 가정하자.

## 리스트를 이용한 데이터프레임 생성
```python
avocados_list = [
    {"date": "2019-11-03", "small_sold": 10376832, "large_sold": 7835071},
    {"date": "2019-11-10", "small_sold": 10717154, "large_sold": 8561348}
]

avocados_2019 = pd.DataFrame(avocados_list)

print(avocados_2019)
```
딕셔너리의 key값이 데이터프레임 생성시 컬럼명이 되고, value값은 해당 컬럼에 해당하는 값이 된다. 하지만 컬럼명을 계속 써줘야 하기 때문에 dictionary of lists를 데이터프레임으로 변경하는게 더욱 효율적이다.

## 딕셔너리를 이용한 데이터프레임 생성
```python
         date  small_sold  large_sold
0  2019-11-03    10376832     7835071
1  2019-11-10    10717154     8561348
```

```python
avocados_dict = {
  "date": ["2019-11-17","2019-12-01"],
  "small_sold": [10859987,9291631],
  "large_sold": [7674135,6238096]
}

avocados_2019 = pd.DataFrame(avocados_dict)

print(avocados_2019)
```

```python
         date  small_sold  large_sold
0  2019-11-17    10859987     7674135
1  2019-12-01     9291631     6238096
```

