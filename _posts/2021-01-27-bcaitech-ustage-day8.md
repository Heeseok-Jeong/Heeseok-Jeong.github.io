---
layout : post
title : Ustage Day 8
subtitle : Pandas|딥러닝 원리 이해
tags : [BoostCamp AI Tech]
author : Heeseok Jeong
comments : True
use_math: True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

<br>

- [Pandas](#pandas)
- [딥러닝 학습방법 이해하기](#딥러닝-학습방법-이해하기)
- [피어 세션](#피어-세션)
- [Today I Felt](#today-i-felt)

<br>

<hr>

<br>

# Pandas

<br>

- panel data
- 구조화된 데이터의 처리를 지원하는 파이썬 라이브러리
- 고성능 계산 라이브러리 numpy 와 통합하여, 강력한 스프레드시트 처리 기능 제공
- 인덱싱, 연산용 함수, 전처리 함수 등 제공, 데이터 처리 및 통계 분석을 위해 사용
- 테이블, 어트리뷰트(피쳐), 인스턴스(튜플), 피쳐벡터(세로열), 데이터(하나의값)가 있음
- 이미지 같은거 처리하는 도구가 아님

### 설치

- 콘다 가상환경에서 conda install pandas 로 설치

### 실행

```python
import pandas as pd
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df_data = pd.read_csv(data_url, sep='\s+', header = None) # sep 은 정규식
df_data.head()
```

- 결과

![ustage_day8_0]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_0.png)

## 기본 사용

## series

- (DataFrame : Data Table 전체를 포함하는 Object)
- Series : DataFrame 중 하나의 Column 에 해당하는 데이터 모음 Object
- 인덱스, 값, 데이터 타입이 나옴, numpy.ndarray 타입
- 딕트도 넣을 수 있음 → 자동으로 키는 인덱스, 밸류는 값이 됨

```python
list_data = [1, 2, 3, 4, 5]

example_obj = pd.Series(data = list_data)
example_obj
'''
0    1
1    2
2    3
3    4
4    5
dtype: int64
'''

dict_data = {'a' : 1, 'b' : 2}
example_obj = pd.Series(dict_data)
example_obj.name = "number"
example_obj.index.name = "alphabet"
example_obj

'''
alphabet
a    1
b    2
Name: number, dtype: int64
'''

example_obj.index
# Index(['a', 'b'], dtype='object', name='alphabet')

# index 값 기준으로 series 생성
dict_data = {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5}
indexes = ['a', 'b', 'c', 'd', 'e', 'f']
series_obj = pd.Series(dict_data, index=indexes)
series_obj
'''
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
f    NaN
dtype: float64
'''
```

## DataFrame

- 시리즈 데이터들의 모임, 테이블이기도 함, 기본 2차원, 주로 csv 나 엑셀 형태로 불러냄
- 컬럼마다 데이터 타입이 다를 수 있음
- 인덱스와 컬럼으로 접근 가능
- df["first_name"] 와 df.first_name 은 같음. fist_name 컬럼 불러옴
- 인덱싱
    - loc - index 이름
    - iloc - index 번호

    ```python
    # 값 중 3 이 나올때까지 추출
    df.loc[:3]
    '''
    first_name	last_name	age	city
    0	Jason	Miller	42	San Francisco
    1	Molly	Jacobson	52	Baltimore
    2	Tina	Ali	36	Miami
    3	Jake	Milner	24	Douglas
    '''

    # 인덱스 3 전까지 추출
    df.iloc[:3]
    '''
    first_name	last_name	age	city
    0	Jason	Miller	42	San Francisco
    1	Molly	Jacobson	52	Baltimore
    2	Tina	Ali	36	Miami
    '''
    ```

- 자세한 코드는 [DataFrame](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pandas/%231/4_pandas_dataframe.ipynb) 에 있습니다.
- boolean index 로 값 설정 가능

```python
df.debt = df.age > 40
df
'''
first_name	last_name	age	city	debt
0	Jason	Miller	42	San Francisco	True
1	Molly	Jacobson	52	Baltimore	True
2	Tina	Ali	36	Miami	False
3	Jake	Milner	24	Douglas	False
4	Amy	Cooze	73	Boston	True
'''
```

- to_csv() : csv 형태로 변환, 파라미터 조절하면 저장도 가능
- 컬럼 삭제
    - del df["debt"] → 아예 테이블에서 삭제해버림
    - df.drop("debt", axis=1) → 리턴해주고 원본은 안바뀜

### Selection & Drop

##### selection : 시리즈(컬럼) 불러오는 작업

- 엑셀 파일 다룰 때 `!conda install --y xlrd` 로 xlrd 설치
- 컬럼명으로 뽑기
    - df["a"] → Series 로 뽑힘, df[["a"]] → DataFrame으로 뽑힘
- 인덱스로 뽑기
    - df["account"][:3] → 어카운트 시리즈의 0, 1, 2 인덱스 지닌 값들 뽑음
    - 인덱스가 영어인데 숫자로 부르면 안 뽑아짐
- fancy index, boolean index 가능
- 인덱스값 설정, df.account = df["account"]
- 데이터 뽑는 방식 (중요)
    1. df["name", "street][:2] # 컬럼과 인덱스 네임
    : 해당 시리즈의 인스턴스 2개 가져옴
    2. df.loc[[211829, 320563], ["name", "street"] # row 네임, col 네임
    : 해당 인덱스에 해당 컬럼 값 가져옴
    3. df.iloc[:2, :2] # 로우 개수, 컬럼 개수
    : 2개 컬럼의 2개 인스턴스 가져옴

- reset_index() : 인덱스 값 초기화 (0, 1, 2 ...), 원본 안바뀜
    - 프로퍼티로 inplace=Ture 하면 원본에 적용
    - 기본적으로 데이터프레임 원본 안건드릴려고 함
- boolean index 로 셀렉션 가능

##### drop : 삭제하는 작업

- df.drop("city", axis=1) # city 컬럼(axis=1) 삭제, 원본 안바뀜

### Series operation

- s1.add(s2) : s1 과 s2 합치는데 겹치는 인덱스가 없는 경우 NaN 으로 채워짐
→ s1 + s2 와 동일

### DataFrame operation

- df1.add(df2, fill_value=0) : 두 데이터 프레임 합치는데 (인덱스 + 컬럼 전부) 안 겹치는 공간은 NaN 으로 채움. fill_value 를 설정하면 NaN 대신 그 값으로 채움

    → df1 + df2 와 기본적으로 동일

### Series + DataFrame

- 데이터프레임의 컬럼과 시리즈의 인덱스가 같으면 브로드캐스팅 연산됨
- add 에서 axis 를 설정하면 broadcasting 실행됨

```python
df = DataFrame(np.arange(16).reshape(4, 4), columns=list("abcd"))
df
'''
a	b	c	d
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15
'''

s = Series(np.arange(10, 14), index=list("abcd"))
s
'''
a    10
b    11
c    12
d    13
dtype: int64
'''

df + s
'''
a	b	c	d
0	10	12	14	16
1	14	16	18	20
2	18	20	22	24
3	22	24	26	28
'''

s2 = Series(np.arange(10, 14))
s2
'''
0    10
1    11
2    12
3    13
dtype: int64
'''

# 시리즈의 인덱스가 없어서 옆으로 붙임, NaN으로 전부 채워짐
df + s2 # == df.add(s2, axis=1)
'''
a	b	c	d	0	1	2	3
0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
'''

df.add(s2, axis=0)
'''
a	b	c	d
0	10	11	12	13
1	15	16	17	18
2	20	21	22	23
3	25	26	27	28
'''
```

## lambda, map, apply

### map for siries

- 판다스의 시리즈 다입의 데이터에도 맵 가능
- function 대신 dict, sequence 형 자료등으로 대체 가능

```python
s1 = Series(np.arange(10))
s1.map(lambda x: x**2).head()
'''
0     0
1     1
2     4
3     9
4    16
dtype: int64
'''

# dict 타입 적용
z = {1: 'a', 2: 'b', 3 : 'c'}
s1.map(z).head()
'''
0    NaN
1      a
2      b
3      c
4    NaN
dtype: object
'''

# DF 타입 적용
s2 = Series(np.arange(10, 20))
s2.map(s2).head()
'''
0   NaN
1   NaN
2   NaN
3   NaN
4   NaN
dtype: float64
'''
```

- replace 처럼 딕트를 써서 데이터프레임의 어떤 값을 키와 매칭하여 밸류로 바꿔줄 수 있음

```python
df["sex_code"] = df.sex.map({"male" : 0, "female" : 1})
df.head()
earn	height	sex	race	ed	age	sex_code
0	79571.299011	73.89	male	white	16	49	0
1	96396.988643	66.23	female	white	16	62	1
2	48710.666947	63.77	female	white	16	33	1
3	80478.096153	63.22	female	other	16	95	1
4	82089.345498	63.08	female	white	17	43	1

df.sex.replace({"male": 0, "female": 1}, inplace=True)
df.head()
earn	height	sex	race	ed	age	sex_code
0	79571.299011	73.89	0	white	16	49	0
1	96396.988643	66.23	1	white	16	62	1
2	48710.666947	63.77	1	white	16	33	1
3	80478.096153	63.22	1	other	16	95	1
4	82089.345498	63.08	1	white	17	43	1
```

### apply

- 맵과 다르게 시리즈 전체에 해당 함수를 적용, 시리즈 단위로 적용시키면 맵과 같은 효과
- 입력 값을 시리즈 데이터로 받아 핸들링 가능
- 내장 연산 함수 (sum, mean, std 등) 를 사용하는 것과 같은 효과

```python
# 셋 다 똑같음
# 함수 정의 후 어플라이
f = lambda x: np.mean(x)
df_info.apply(f)

# 내장 함수 어플라이
df_info.apply(np.mean)

# 내장 함수 사용
df_info.mean()

'''
earn      32446.292622
height       66.592640
age          45.328499
dtype: float64
'''
```

## pandas built-in function

### describe

- 가지고 있는 값 중 Numeric type (숫자들) 의 요약 정보를 보여줌

### unique

- 시리즈 데이터의 유일한 값을 리스트로 반환함

### sum

- 축을 기준으로 연산 진행
- sub, mean, min ...

### isnull

- 테이블에서 값이 NaN(null) 인 데이터는 True 로 아니면 False 로 만들어서 DF 반환
- isnull().sum() 하면 컬럼에 관해서 널값 개수 세어 줌

### sort_values

- 컬럼 값을 기준으로 정렬해줌
- df.sort_values(["age", "earn"], ascending=True).head()

### corr, conv, corrwith

- corr : 상관계수 구해줌, 아규먼트로 시리즈 넣어도 되고 안넣으면 테이블 전체로 구해줌
- cov : 공분산 구해줌
- corrwith : 테이블과 아규먼트의 상관계수 구해줌
- 위 작업들 할 때 스트링 값이면 난감하므로 라벨링 (0, 1) 등 으로 바꿔서 사용하면 용이

### 여러개 컬럼 다룰 때

- 컬럼에 대한 리스트를 만들어서 수행

### 한 번에 볼 수 있는 양 조절

- pd.options.display.max_rows = 100

<br>

<hr>

<br>

# 딥러닝 학습방법 이해하기

<br>

- 신경망은 기본적으로 비선형 모델, 하지만 뜯어보면 선형모델과 비선형 함수들의 결합으로 이루어짐
- 비선형 모델을 어떻게 선형 모델로 만드는지 알 것

### 선형 모델

- O(nxp) = X(nxd)W(dxp) + b(nxp)
    - X : 데이터 행렬, W : X 각 데이터를 다른 공간으로 보내주는 가중치 행렬
    - b 는 각 행들이 같은 값을 지님 (y 절편 벡터)
- 아래 그림과 같이 각 데이터 x 는 각 결과 o 에 대해 화살표로 연결됨
→ 화살표는 x 를 o 로 연결하는 W (웨이트) 라고 생각하면 됨
→ d 개가 p 개로 가므로 W 는 dxp 개가 있음
    - W 는 d 개의 변수로 p 개의 선형모델을 만들어서 p 개의 잠재변수를 설명하는 모델

![ustage_day8_1]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_1.png)

## Softmax

- 분류 문제에서 모델 (비선형 모델) 의 출력을 확률로 해석할 수 있게 변환해주는 연산, 데이터가 어떤 클래스에 속하는지 분류
- 분류 문제에서 선형모델과 소프트맥스 함수 결합하여 예측

```python
def softmax(vec):
	denumerator = np.exp(vec = np.max(vec, axis=-1, keepdims=True)
	numerator = np.sum(denumerator, axis=-1, keepdims=True)
	val = denumerator / numerator
	return val
```

- 학습에는 소프트맥스가 필요하지만, 추론(실전) 할 때는 소프트맥스 말고 one_hot 벡터로 (최대값을 가진 주소만 1 로 출력하는 연산) 사용

## 신경망

- 선형모델과 활성함수를 합성한 함수 = 히든벡터
- 활성함수와 소프트맥스 차이 : 소프트맥스는 출력물 모든 값 고려해서 출력 - 벡터를 받음, 활성함수는 하나의 주소만 고려해서 출력 - 실수값 1 개만 받음

### 활성함수

- 실수를 받아서 실수를 반환
- 활성함수를 쓰지 않으면 선형모델임, 활성함수 써야 비선형으로 바뀜
- 시그모이드나 tanh 를 예전에 썼지만, 이제는 **ReLU** 함수 사용
- z 행렬의 모든 원소에 활성함수 씌움 (실수를 받기 때문에)

### 신경망 수식

![ustage_day8_2]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_2.png)

- 2-layer 신경망 : x → z 에서 W1, 활성함수(시그마)(z) → o 에서 W2 사용

    ![ustage_day8_3]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_3.png)

- 위 구조를 반복적으로 사용하면 multi-layer perceptron, MLP 신경망이 됨
- 위 작업은 Forward propagation!
- 왜 층을 여러개 쌓나요?
    - 층이 깊을수록 목적함수를 근사하는데 필요한 뉴런(노드) 의 숫자가 훨씬 빨리 줄어들어 더 효율적인 학습 가능 → 층이 얇으면 넓은 신경망이 돼야함
    - 층이 깊으면 복잡한 함수 표현은 가능하지만, 최적화는 더 어려워짐 (나중에 convolution 에서 residual 배울 것)

## 딥러닝 학습원리: 역전파 알고리즘

- 순전파는 선형모델에 활성함수 썼었음
- 역전파는 학습에서 경사하강법을 써야하는데 이 때 가중치를 업데이트함에 있어 합성함수를 다뤄서 최초의 가중치를 구할 수 있게 해줌
- 윗 층의 그레디언트를 구하여 다음층, 다음층 역순으로 전달함

![ustage_day8_4]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_4.png)

### 역전파 원리 이해하기

- 합성함수 미분법인 연쇄법칙(체인룰) 기반 자동미분 사용

![ustage_day8_5]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_5.png)

→ 모든 변수에 대해 편미분을 저장해야함 → 순전파보다 메모리 많이 사용

- 예제

![ustage_day8_6]({{ site.baseurl }}/assets/img/ustage_day8/ustage_day8_6.png)

- 파란색 : 순전파, 빨간색 : 역전파
- 역전파 결과값부터해서 W1까지 연쇄법칙 사용

### Further Question

1. 분류 문제에서 softmax 함수가 사용되는 이유가 뭘까요?
2. softmax 함수의 결과값을 분류 모델의 학습에 어떤식으로 사용할 수 있을까요?

### 교수님 말씀

- 딥러닝 원리 (여러 층의 선형모델과 활성함수에 대한 함성함수, 합성함수의 그래디언트 계산을 위해 연쇄법칙 적용한 역전파 알고리즘 사용) 이해할 것

<br>

<hr>

<br>

# 피어 세션

<br>

## 세미나

### 후미님 - 마르하바!

- 파병을 다녀옴 (동명부대 레바논, 베이루트)
- 8개월

### 원딜님

- 기초 통계 (순한맛)
    - 확률
        - 어떤 시행을 반복할 때 N 번 시행에서 사건 A 가 발생한 횟수를 n(A) 라 하면 확률은 무한히 발생할 때 P(A) = n(A)/N
        - 빈도론 학파
        - 베이지안 학파
    - 확률의 공리 3 조건 무조건 만족해야 확률
    - 조건부 확률
    > P(A\|B) = P(A교B)/P(B)
    - 확률 변수 : 표본공간의 사건 또는 원소를 실수값으로 변환하는 함수
        - 이산형 변수
        - 연속형 변수
    - 이산분포
        - 베르누이 분포
        - 포아손 분포 등
    - 연속분포
    - 가능도 함수 : 간단하게는 확률밀도함수랑 비슷, 가능도함수는 확률함수가 아님

## 피어세션 피드백

- 이슈는 퍼블릭 레포지토리, 코드 리뷰는 프라이빗 레포지토리
- 개인 발표 자료 등은 구글드라이브에 자율적으로 올리기로

<br>

<hr>

<br>

# Today I Felt

<br>

## 꿈틀꿈틀

오늘은 아니지만,, 어제 행렬의 미분과 목적식을 이용한 경사하강법에 대한 개념이 꼬여 밥먹는 시간 빼고 밤까지 계속 공부했다 (끈기 점수 +1). 한 번 꼬이니까 더이상 진전이 없어 다시 월요일 강의부터 공부했고 그 끝에 이해할 수 있었다. 그리고 안 까먹기 위해 자기 전에 누워서 다시 상기했다. 비록 느리게 걸어가도 이 마음가짐으로 캠프가 끝나고도 계속 발전하는 사람이 되고싶다.
