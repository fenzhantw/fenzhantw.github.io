---
title: "[pandas] pandas 데이터 핸들링 정리(3) Index"
excerpt: "set_index,reset_index,multi-level index,sort_index，"
categories:
    - pandas

tag:
    - python
    - pandas2

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---

## Intro
Pandas는 다양한 메서드와 속성이 존재하기 때문에 정리를 하지 않으면 헷갈릴 수 있으므로, 배운 내용을 정리하고자 합니다

그리고 pandas는 원하는 데이터 핸들링을 하기 위해서 하나의 방법만 존재하지 않기 때문에, 자신만의 판다스 사용 방법을 외워서 숙달한다면 더 빠르게 실전 상황에서 데이터 핸들링을 할 수 있을 겁니다.

이번에는 데이터 핸들링 첫번째로, drop duplicate, value_counts, 요약된 통계량을 보여주는 summary statistics와 groupby를 정리하겠습니다

## 데이터 소개

<img width="438" alt="캡처" src="https://user-images.githubusercontent.com/81638919/132085866-f3f3c23f-1613-4bf9-85bb-42ab386d4788.PNG">


## set_index
```
temperatures_ind = temperatures.set_index("city")

# Look at temperatures_ind
print(temperatures_ind)
```

```
              date        country  avg_temp_c
city                                         
Abidjan 2000-01-01  Côte D'Ivoire      27.293
Abidjan 2000-02-01  Côte D'Ivoire      27.685
Abidjan 2000-03-01  Côte D'Ivoire      29.061
Abidjan 2000-04-01  Côte D'Ivoire      28.162
Abidjan 2000-05-01  Côte D'Ivoire      27.547
...            ...            ...         ...
Xian    2013-05-01          China      18.979
Xian    2013-06-01          China      23.522
Xian    2013-07-01          China      25.251
Xian    2013-08-01          China      24.528
Xian    2013-09-01          China         NaN


```


## reset_index

```
# Reset the index, keeping its contents
print(temperatures_ind.reset_index())

```
```
          city       date        country  avg_temp_c
0      Abidjan 2000-01-01  Côte D'Ivoire      27.293
1      Abidjan 2000-02-01  Côte D'Ivoire      27.685
2      Abidjan 2000-03-01  Côte D'Ivoire      29.061
3      Abidjan 2000-04-01  Côte D'Ivoire      28.162
4      Abidjan 2000-05-01  Côte D'Ivoire      27.547
...        ...        ...            ...         ...
16495     Xian 2013-05-01          China      18.979
16496     Xian 2013-06-01          China      23.522
16497     Xian 2013-07-01          China      25.251
16498     Xian 2013-08-01          China      24.528
16499     Xian 2013-09-01          China         NaN
```

```
# Reset the index, dropping its contents
print(temperatures_ind.reset_index(drop=True))

```
```
            date        country  avg_temp_c
0     2000-01-01  Côte D'Ivoire      27.293
1     2000-02-01  Côte D'Ivoire      27.685
2     2000-03-01  Côte D'Ivoire      29.061
3     2000-04-01  Côte D'Ivoire      28.162
4     2000-05-01  Côte D'Ivoire      27.547
...          ...            ...         ...
16495 2013-05-01          China      18.979
16496 2013-06-01          China      23.522
16497 2013-07-01          China      25.251
16498 2013-08-01          China      24.528
16499 2013-09-01          China         NaN

```

## 인덱싱과 loc 활용

```

# Make a list of cities to subset on
cities = ["Moscow", "Saint Petersburg"]

# Subset temperatures using square brackets
print(temperatures[temperatures['city'].isin(cities)])

# # Subset temperatures_ind using .loc[]
print(temperatures_ind.loc[cities])

```

```
                       date country  avg_temp
city                                           
Moscow           2000-01-01  Russia      -7.313
Moscow           2000-02-01  Russia      -3.551
Moscow           2000-03-01  Russia      -1.661
Moscow           2000-04-01  Russia      10.096
Moscow           2000-05-01  Russia      10.357
...                     ...     ...         ...
Saint Petersburg 2013-05-01  Russia      12.355
Saint Petersburg 2013-06-01  Russia      17.185
Saint Petersburg 2013-07-01  Russia      17.234
Saint Petersburg 2013-08-01  Russia      17.153
Saint Petersburg 2013-09-01  Russia         NaN

```
## setting multi-level indexex


```

# Index temperatures by country & city
temperatures_ind = temperatures.set_index(['country','city'])

# List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
rows_to_keep = [("Brazil","Rio De Janeiro"), ("Pakistan","Lahore")]

# Subset for rows to keep
print(temperatures_ind.loc[rows_to_keep])
```
```
                              date  avg_temp_c
country  city                                 
Brazil   Rio De Janeiro 2000-01-01      25.974
         Rio De Janeiro 2000-02-01      26.699
         Rio De Janeiro 2000-03-01      26.270
         Rio De Janeiro 2000-04-01      25.750
         Rio De Janeiro 2000-05-01      24.356
...                            ...         ...
Pakistan Lahore         2013-05-01      33.457
         Lahore         2013-06-01      34.456
         Lahore         2013-07-01      33.279
         Lahore         2013-08-01      31.511
         Lahore         2013-09-01         NaN
```
    
```
# Sort temperatures_ind by index values
print(temperatures_ind.sort_index())

# Sort temperatures_ind by index values at the city level
print(temperatures_ind.sort_index(level="city"))

# Sort temperatures_ind by country then descending city
print(temperatures_ind.sort_index(level=["country", "city"], ascending = [True, False]))
```
```
                         date  avg_temp_c
country     city                         
Afghanistan Kabul  2000-01-01       3.326
            Kabul  2000-02-01       3.454
            Kabul  2000-03-01       9.612
            Kabul  2000-04-01      17.925
            Kabul  2000-05-01      24.658
...                       ...         ...
Zimbabwe    Harare 2013-05-01      18.298
            Harare 2013-06-01      17.020
            Harare 2013-07-01      16.299
            Harare 2013-08-01      19.232
            Harare 2013-09-01         NaN
```
```
# Sort the index of temperatures_ind
temperatures_srt = temperatures_ind.sort_index()

# Subset rows from Pakistan to Russia
print(temperatures_srt.loc["Pakistan":"Russia"])
```
```
                                date  avg_temp_c
country  city                                   
Pakistan Faisalabad       2000-01-01      12.792
         Faisalabad       2000-02-01      14.339
         Faisalabad       2000-03-01      20.309
         Faisalabad       2000-04-01      29.072
         Faisalabad       2000-05-01      34.845
...                              ...         ...
Russia   Saint Petersburg 2013-05-01      12.355
         Saint Petersburg 2013-06-01      17.185
         Saint Petersburg 2013-07-01      17.234
         Saint Petersburg 2013-08-01      17.153
         Saint Petersburg 2013-09-01         NaN
```
```
# Try to subset rows from Lahore to Moscow
print(temperatures_srt.loc[("Pakistan","Lahore"):("Russia","Moscow")])
```
```
                     date  avg_temp_c
country  city                         
Pakistan Lahore 2000-01-01      12.792
         Lahore 2000-02-01      14.339
         Lahore 2000-03-01      20.309
         Lahore 2000-04-01      29.072
         Lahore 2000-05-01      34.845
...                    ...         ...
Russia   Moscow 2013-05-01      16.152
         Moscow 2013-06-01      18.718
         Moscow 2013-07-01      18.136
         Moscow 2013-08-01      17.485
         Moscow 2013-09-01         NaN
```
