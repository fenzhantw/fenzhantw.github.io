---
title: "[python] pandas 데이터 핸들링 정리(2)"
excerpt: "Drop Duplicate, value_counts,summary statistics,groupby"
categories:
    - python

tag:
    - python
    - pandas

author_profile: true    #작성자 프로필 출력 여부

toc: true   #Table Of Contents 목차 
toc_sticky: true
---

## Intro
Pandas는 다양한 메서드와 속성이 존재하기 때문에 정리를 하지 않으면 헷갈릴 수 있으므로, 배운 내용을 정리하고자 합니다.
그리고 pandas는 원하는 데이터 핸들링을 하기 위해서 하나의 방법만 존재하지 않기 때문에, 자신만의 판다스 사용 방법을 외워서 숙달한다면 더 빠르게 실전 상황에서 데이터 핸들링을 할 수 있을 겁니다. 
이번에는 데이터 핸들링 첫번째로, drop duplicate, value_counts, 요약된 통계량을 보여주는 summary statistics와 groupby를 정리하겠습니다.

## 데이터 소개

```
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S

```

## Drop duplicate
```
# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(['store','type'])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(['store','department'])
print(store_depts.head())

# Subset the rows where is_holiday is True and drop duplicate dates
holiday_dates = sales[sales['is_holiday']==True].drop_duplicates('date')

# Print date col of holiday_dates
print(holiday_dates.columns)

```
* 상품명과 판매개수를 구하려는 쿼리를 실행하기 위해 간단한 LEFT JOIN으로 테이블을 JOIN하고 쿼리를 실행시킨다.
* 하지만 다음 쿼리 실행시 COUNT로 집계가 되지 않는 상품의 경우 NULL 값이 출력되지 않고, COUNT(A.id) = 1인 값부터 출력이 된다.

## Value counts

```
# Count the number of stores of each type
store_counts = store_types["type"].value_counts()
print(store_counts)

# Get the proportion of stores of each type
store_props = store_types["type"].value_counts(normalize=True)
print(store_props)

# Count the number of each department number and sort
dept_counts_sorted = store_depts["department"].value_counts(sort=True)
print(dept_counts_sorted)

# Get the proportion of departments of each number and sort
dept_props_sorted = store_depts["department"].value_counts(sort=True, normalize=True)
print(dept_props_sorted)
```

## Group by

```
# From previous step
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Group by type and is_holiday; calc total weekly sales
sales_by_type_is_holiday = sales.groupby(["type","is_holiday"])["weekly_sales"].sum()
print(sales_by_type_is_holiday)

```
# Import numpy with the alias np
import numpy as np

# For each store type, aggregate weekly_sales: get min, max, mean, and median
sales_stats = sales.groupby("type")["weekly_sales"].agg([np.min,np.max,np.mean,np.median])

# Print sales_stats
print(sales_stats)

# For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
unemp_fuel_stats = sales.groupby("type")['unemployment','fuel_price_usd_per_l'].agg([np.min,np.max,np.mean,np.median])

# Print unemp_fuel_stats
print(unemp_fuel_stats)

```

## pivot tables

Pivot tables are the standard way of aggregating data in spreadsheets. In pandas, pivot tables are essentially just another way of performing grouped calculations
That is, the .pivot_table() method is just an alternative to .groupby().

```
dogs.groupby(["color","breed"])["weight_kg"].mean()
dog.pivot_table(values="weight_kg",index="color",columns="breed",fill_value=0,margin=True(요약통계))

sales.pivot_tables(values="weekly_sales",index="type")

          weekly_sales
    type              
    A        23674.667
    B        25696.678
    
# Import NumPy as np
import numpy as np

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(values="weekly_sales",index="type",aggfunc=[np.median])

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)

          median
     weekly_sales
type             
A        11943.92
B        13336.08

# Pivot for mean weekly_sales by store type and holiday 
mean_sales_by_type_holiday = sales.pivot_table(values="weekly_sales",index=["type"],columns="is_holiday")

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)

is_holiday      False    True 
type                          
A           23768.584  590.045
B           25751.981  810.705

print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0,margins=True))
```

