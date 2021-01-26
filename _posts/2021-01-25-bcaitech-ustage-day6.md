---
layout : post
title : Ustage Day 6
subtitle : 파이썬 Numpy 와 벡터, 행렬
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

- [Numpy](#numpy)
- [벡터가 뭐에요?](#벡터가-뭐에요)
- [행렬가 뭐에요?](#행렬이-뭐에요)
- [피어 세션](#피어-세션)
- [Today I Felt](#today-i-felt)

<br>

<hr>

<br>

# Numpy

- Numerical Python, 파이썬의 고성능 과학 계산용 패키지
- 왜 사용하는가?
    - 행렬과 매트릭스를 코드로 표현하는 것을 지원
    - 다양한 매트릭스 계산을 지원
    - 큰 매트릭스에 대한 표현 지원
    - 처리 속도 문제
- 특징
    - 일반 리스트에 비해 빠르고, 메모리 효율적
    - 반복문 없이 데이터 배열에 대한 처리 지원
    - 선형대수와 관련된 다양한 기능 제공
    - C, C++, 포트란 언어와 통합 가능
- 사용
    1. 가상환경 실행
    > activate base
    2. `numpy` 설치
    > conda install numpy
    3. 쥬피터 실행
    > jupyter notebook
    4. numpy 임포트 `import numpy as np`

<br>

## ndarray

- np.array 함수로 ndarray (배열) 생성

    ```python
    import numpy as np

    np.array([1, 4, 5], float)
    print(test_array)
    # [1. 4. 5.]
    type(test_array[3])
    # numpy.float64
    ```

- 리스트와 다르게 하나의 데이터 type 만 배열에 넣을 수 있고, dynamic typing 을 지원하지 않음
- C 의 Array 를 사용하여 배열 생성
- 파이썬 리스트와 차이
    - 넘파이 어레이는 객체의 데이터 영역이 값을 지닌 배열을 직접 가리킴 + 각 공간 크기 일정함
    - 파이썬 리스트는 객체의 아이템 영역이 값을 가리키는 메모리 주소를 가리킴
    - 두 개의 ndarray 가 같은 값을 지녀도 두 원소에 대해 `a[0] is b[0]` 를 수행하면 False 가 나옴 (리스트에서는 True)
- shape : ndarray 의 dimension 구성을 tuple 로 반환
    - array 의 RANK 에 따라서 불리는 이름
        - rank 0, ex) 7 : scalar
        - rank 1, ex) [10, 10] : vector ⇒ (2,) ⇒ col
        - rank 2, ex) [[1, 2], [3, 4]] : matrix ⇒ (2, 2) ⇒ row, col
        - rank 3, ex) [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] : 3-tensor ⇒ (2, 2, 2) ⇒ depth, row, col
        - rank n : n-tensor
    - ndim : 차원 개수
    - size : 데이터 개수
- dtype : ndarray 의 데이터 타입을 반환
    - C 의 data type 과 compatible
    - 주로 float64 사용
    - nbytes : ndarray 객체의 메모리 크기를 반환

<br>

## shape 다루기

- reshape : 어레이의 shape 크기를 변경함, element 개수는 동일
    - -1 사용하면 나머지 엘리먼트를 보고 알아서 계산해줌
    - 원본은 바뀌지 않음

```python
import numpy as np

test_matrix = [[1, 2, 3, 4], [1, 2, 5, 8]]
np.array(test_matrix).shape
# (2, 4)

np.array(test_matrix).reshape(4, 2)
# array([[1, 2],
#        [3, 4],
#        [1, 2],
#        [5, 8]])

np.array(test_matrix).reshape(2, 2, 2)
# array([[[1, 2],
#         [3, 4]],

#        [[1, 2],
#         [5, 8]]])

np.array(test_matrix).reshape(-1, 1)
# -1 => 8
# array([[1],
#        [2],
#        [3],
#        [4],
#        [1],
#        [2],
#        [5],
#        [8]])
```

- flatten : 다차원 array 를 1 차원 array 로 변환, reshape(1, -1) 은 원래  디멘션이 유지되는 반면, flatten 은 스칼라로 바꿔버림

<br>

## Indexing & Slicing

### Indexing

- 리스트와 달리 이차원 배열에서 [0, 0] 표기법 제공
    - a[0, 0] == a[0][0]
- 매트릭스일 경우 앞은 row, 뒤는 col 의미

### Slicing

- 리스트와 달리 행과 열 부분을 나눠서 슬라이싱 가능
- 매트릭스의 부분 집합을 추출할 때 유용

```python
a
# array([[10.,  2.,  3.,  4.],
#        [ 1.,  2.,  5.,  8.]])

a[:, 2:]
# array([[3., 4.],
#        [5., 8.]])

a[1, 1:3]
# array([2., 5.])

a[1:2]
# array([[1., 2., 5., 8.]])
```

- `::` 는 step 기능, x 칸 뛰어넘음

<br>

## Creation Function

### arange

- 범위를 지정하여, 값의 리스트를 생성하는 명령어

```python
b = np.arange(30)
b
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

b = b.reshape(5, 6)
b
# array([[ 0,  1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10, 11],
#        [12, 13, 14, 15, 16, 17],
#        [18, 19, 20, 21, 22, 23],
#        [24, 25, 26, 27, 28, 29]])

np.arange(0, 10, 0.5)
# array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
#        6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])
```

### ones, zeros and empty

- ones : 1 로 가득찬 array 생성
- zeros : 0 "
- empty : shape 만 주어지고 비어있는 array 생성 (memory initialization 되지 않음, 값 초기화를 안해서 메모리에 들어있는 기존 값 나옴)
- something_like (ones_like) : 기존 어레이의 shape 만큼 1, 0, 또는 empty 어레이 반환

### identity, eyes, diag

- identity : 단위 행렬 (i 행렬) 생성
- eyes : 대각선이 1인 행렬 생성, 시작점 k 변경 가능
- diag : 대각 행렬의 값을 추출함

```python
np.identity(n = 3, dtype = np.int8)
# or
np.identity(3)
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])

np.eye(3)
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])

np.eye(3, 5, k=2)
# array([[0., 0., 1., 0., 0.],
#        [0., 0., 0., 1., 0.],
#        [0., 0., 0., 0., 1.]])

b
# array([[ 0,  1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10, 11],
#        [12, 13, 14, 15, 16, 17],
#        [18, 19, 20, 21, 22, 23],
#        [24, 25, 26, 27, 28, 29]])
np.diag(b)
# array([ 0,  7, 14, 21, 28])
```

### random sampling

- 데이터 분포에 따른 샘플링으로 어레이 생성
- uniform (균등분포), normal (랜덤분포), exponential (지수분포)

```python
np.random.normal(0, 1, 10)
# array([-0.01407373, -0.90342127,  3.04201844,  2.45111633, -0.32714641,
#        -0.07848476, -0.65410073,  0.36465804, -0.26742882,  0.84662701])

np.random.normal(0, 1, 10).reshape(2, 5)
# array([[ 0.10503633,  0.711939  ,  1.62220759,  0.01422367, -1.71976678],
#        [-2.26470677,  0.57157168, -0.88920551, -0.00722432,  0.49078863]])

np.random.exponential(1)
# 1.2748717265749823

np.random.exponential(scale=2, size=10)
# array([0.35423572, 0.71810005, 0.79145404, 3.74830117, 1.94590341,
#        2.61224156, 4.21561447, 0.86994613, 1.01113432, 2.48008649])
```

<br>

## Opretaion Functions

- sum : 어레이 모든 값 더함
- mean, std, var 등 매우 많은 연산이 지원됨
- 어떻게 연산을 지원하는가?
    - axis : 모든 operation function 실행할 때 기준이 되는 dimension 축
    - sum 할 때 매트릭스에서 axis 를 0 으로 하면 같은 col 끼리 더함

```python
test_array = np.arange(1, 13).reshape(3, 4)
test_array
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

test_array.sum(axis=1)
# array([10, 26, 42])

test_array.sum(axis=0)
# array([15, 18, 21, 24])
```

<br>

## concatenate

- 어레이를 합치는 (붙이는) 함수
- vstack : 두 가로 벡터를 붙임
- hstack : 두 세로 벡터 (사실은 2 차원 어레이) 를 붙임
- concatenate, axis 를 설정하여 사용

### newaxis

- 값은 그대로 두고 축을 추가해줌

<br>

## array operations

- numpy 는 어레이간 기본적인 사칙 연산 지원함 (Element-wise operation, 어레이간 shape 가 같을 때 일어나는 연산)

### Dot product

- dot 함수 사용

### transpose

- transpose 또는 T 함수 사용

### broadcasting

- Element-wise operation 과 차이를 기억하자
- shape 이 다른 배열 간 연산을 지원하는 기능
- 퍼져나가는 형식으로 연산이 진행됨
- matrix, scalar 사이 또는 matrix, vector 간 연산에도 지원

```python
# matrix, scalar
test_array
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

test_array + 100
# array([[101, 102, 103, 104],
#        [105, 106, 107, 108],
#        [109, 110, 111, 112]])

# matrix, vector
test_matrix = np.arange(1, 13).reshape(4,3)
test_vector = np.arange(10, 40, 10)
test_matrix + test_vector
# array([[11, 22, 33],
#        [14, 25, 36],
#        [17, 28, 39],
#        [20, 31, 42]])
```

<br>

## numpy performance

### timeit

- 쥬피터 환경에서 코드의 퍼포먼스를 체크하는 함수

```python
iternation_max = 1e9
scalar = 2

%timeit np.arange(iternation_max) * scalar # numpy 를 이용한 성능 측정
# 51.4 µs ± 484 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

### 속도 차이

- 일반적으로 속도는 for loop < list comprehension < numpy
- 1 억번의 루프가 돌 때, 약 4 배 이상 차이를 보임
- 넘파이는 C 로 구현되어 성능을 확보하는 대신 동적 타이핑 포기함
- 대용량 계산에서 넘파이 사용
- Concatenate 처럼 계산이 아닌 할당 연산에서는 속도 차이가 없음

<br>

## Comparisions

- 어레이들끼리 비교

### All 과 Any

- all : 어레이의 모든 원소에 대해 조건 만족하는지 확인 후 True or False 를 반환, and 기능
- any : 어레이의 원소 중 하나라도 조건 만족하는지 확인 후 True or False 를 반환, or 기능

### logical_operation

- 두 어레이의 각 원소에 대해 and, or, not 등의 연산 수행하여 각 원소마다 체크한 어레이 반환

### np.where

- where(condition) → 조건 만족하는 원소의 인덱스 찾아줌
- where(condition, TRUE, FALSE) → 각 원소에 대해 조건 만족하면 true 에 넣은 밸류, 아니면 false 에 넣은 밸류 어레이에 담아서 리턴

```python
a = np.arange(5, 15)
a
# array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

np.where(a > 10)
# (array([6, 7, 8, 9]),)
```

### isnan, isinfinite

- 어레이 내장함수로 체크하는데 사용

### argmax & argmin

- 어레이 내 최댓값, 최솟값 인덱스 반환
- axis 기반의 반환도 가능

### argsort

- 어레이에서 작은 값 기준으로 정렬하여 인덱스 어레이 반환

## boolean & fancy index

### boolan index

- 쿼리처럼 특정 조건에 따른 값을 배열 형태로 추출
- 비교 연산 함수들도 모두 사용가능
- 조건 맞는 애들의 인덱스를 반환, 원래 값이랑 아규먼트랑 어레이 크기(쉐이프) 같아야함

```python
test_array = np.array([1, 4, 0, 2, 3, 8, 9, 7], float)
test_array > 3
# array([False,  True, False, False, False,  True,  True,  True])

# boolean index
test_array[test_array > 3]
# array([4., 8., 9., 7.])
```

### fancy index

- boolean 대신 인덱스 값을 넣어줌, 인테져 리스트를 씀
- 어레이를 인덱스 밸류로 사용해서 값 추출
- take 함수와 동일

```python
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 int 로 선언
a[b]
# array([2., 2., 4., 8., 6., 4.])

a.take(b)
# array([2., 2., 4., 8., 6., 4.])
```

- 매트릭스 형태의 데이터에도 적용가능

```python
a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
a[b, c]
# array([ 1.,  4., 16., 16.,  4.])
a[b]
# array([[ 1.,  4.],
#        [ 1.,  4.],
#        [ 9., 16.],
#        [ 9., 16.],
#        [ 1.,  4.]])
```

<br>

## numpy data I/O

### loadtxt & savetxt

- text type 데이터를 읽고 저장하는 기능

### save

- txt 말고 numpy object 를 피클형태로 저장, 확장자 npy

<br>

<hr>

<br>

# 벡터가 뭐에요?

## 벡터

- 벡터는 숫자를 원소로 가지는 리스트 또는 배열
- 차원이 존재
- 벡터는 공간에서 한 점을 나타냄
    - 1 차원 [x], 2 차원 [x, y], 3 차원 [x, y, z] ...
- 원점으로부터 상대적 위치를 표현함
- 벡터에 숫자를 곱하면 방향은 그대로고 길이만 변함 → 스칼라곱
- 스칼라곱에서 스칼라가 음수이면 반대 방향으로 진행됨
- 같은 모양을 가진 벡터끼리 덧셈, 뺄셈, 성분곱 (elementry-wise) 가능
- 두 벡터의 덧셈은 다른 벡터로부터 상대적 위치 이동을 표현

### 벡터의 노름

- 벡터의 노름은 원점에서부터의 거리, 임의의 차원 d 에 대해 성립
- 여러 노름 공식이 있지만 두 가지 소개

$L_1$ 노름은 각 성분의 변화량의 절대값을 모두 더함

```python
def l1_norm(x):
	x_norm = np.abs(x)
	x_norm = np.sum(x_norm)
	return x_norm
```

$L_2$ 노름은 피타고라스 정리를 이용해 유클리드 거리를 계산, np.linalg.norm 으로 구현 가능

```python
def l2_norm(x):
	x_norm = x*x
	x_norm = np.sum(x_norm)
	x_norm = np.sqrt(x_norm)
	return x_norm
```

- 노름의 종류에 따라 기하학적 성질이 달라짐
    - L1 노름 상의 길이가 1 인 원은 마름모
        - Robust 학습, Lasso 회귀 등에 사용
    - L2 노름 상의 길이가 1 인 원은 원
        - Laplace 근사, Ridge 회귀 등에 사용

### 두 벡터 사이의 거리

- 두 점 사이의 거리 구하는 것과 동일, L1, L2 노름을 이용해 계산할 수 있음
- 벡터의 뺄셈 이용!
- `||x - y||`

### 두 벡터 사의 각도 구하기

- L2 노름으로 구할 수 있음
- 제 2 코사인 법칙 사용
- 내적 (np.inner) 을 사용하여 분자를 쉽게 구할 수 있음

```python
def angle(x, y):
	v = np.inner(x, y) / (l2_norm(x) * l2_norm(y))
	theta = np.arccos(v)
	return theta
```

### 내적은 어떻게 해석할까?

- 내적은 정사영(orthogonal projection) 된 벡터의 길이와 관련 있다.
- Proj(x) 의 길이는 코사인법칙에 의해 \|\|x\|\|cosθ 가 된다.
- 내적은 정사영의 길이를 벡터 y 의 길이 \|\|y\|\| 만큼 조정한 값이다.
- 내적은 두 벡터의 유사도를 측정하는데 사용 가능 

<br>

<hr>

<br>

# 행렬이 뭐에요?

## 행렬

- 벡터를 원소로 가지는 2 차원 배열
- numpy 는 행벡터를 기본 단위로 가짐
- n x m 행렬 : m 개의 원소지닌 벡터를 n 개 가진 행렬
- **X** = $(x_\mathit{ij})$ 로 표기하기도 함

### 전치 행렬

- $X^T = (x_\mathit{ji})$, Transpose of X
- n x m 행렬 → m x n 행렬
- 행렬의 연산을 위해 많이 사용!
- 벡터에도 적용 가능. 행벡터 → 열벡터

### 행렬의 이해 (1)

- 벡터가 공간에서 한 점을 의미한다면 **행렬은 여러 점**들을 의미
- 행렬의 행벡터 $x_i$  는 i 번째 데이터를 의미
- 행렬의 $x_\mathit{ij}$ 는 i 번째 데이터의 j 번째 변수 값

### 행렬의 덧셈, 뺄셈, 성분곱, 스칼라곱

- 같은 모양을 가진 행렬끼리 덧셈, 뺄셈, 성분곱 가능 (element-wise)

### 행렬 곱셈

- maxtrix multiplication : i 번째 행벡터와 j 번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 계산
- $XY = (\sum_k{x_\mathit{ik}y_\mathit{kj}})$

### 행렬의 내적

- 일반 수학에서는 $X^TY$ 이지만, numpy 의 `np.inner` 는 행벡터 기준이므로 $XY^T$ (중요)
- $XY^T = (\sum_k{x_\mathit{ik}y_\mathit{jk}})$

⇒ *행렬곱에서는 X 의 열의 개수와 Y 의 행의 개수가 같아야하고, 행렬의 내적에서는 X 의 열의 개수 (행벡터 크기) 와 Y 의 열의 개수가 같아야 함! 

### 행렬의 이해 (2)

- 데이터를 저장하는게 아니라, 서로 다른 **두 데이터를 연결**시키는 연산자로 이해 할 수 있음
- 행렬곱을 통해 **벡터**를 **다른 차원의 공간으로 보낼** 수 있음
    - m x n 행렬과 n x 1 벡터를 곱하면 벡터는 n → m 차원으로 이동이 됨 (Linear Transform)
- 행렬곱을 통해 **패턴을 추출**하거나 **데이터를 압축**할 수 있음

<br>

## 역행렬

- 어떤 행렬 A 의 연산을 거꾸로 되돌리는 행렬을 역행렬 (inverse matrix) 라고 부르고 $A^\mathit{-1}$ 라고 표기함
- 역행렬은 행과 열 숫자가 같고 행렬식 (detA) 가 0 이 아닌 경우에만 가능
- $AA^\mathit{-1} = A^\mathit{-1}A = I$
- `np.linalg.inv(X)`

#### 유사 역행렬

- 무어-펜로즈 역행렬 (유사역행렬) 은 행과 열의 숫자가 달라도 역행렬 구할 수 있음
- $A^+$ 로 표현
- 행의 개수가 열의 개수보다 많은 경우 $A^+ = (A^TA)^\mathit{-1}A^T$
- 열의 개수가 행의 개수보다 많은 경운 $A^+ = A^T(AA^T)^\mathit{-1}$
- $A^+A = I$ 성립
- `np.linalg.pinv(Y)`
- 조심할 점
    - 행이 열보다 많은 경우, $A^+A = I$ 만 성립
    - 열이 행보다 많은 경우, $AA^+ = I$ 만 성립
- 응용 1 : 변수 개수와 식의 개수가 같지 않은 연립방정식 풀기
⇒ Ax = b 에서 양 변의 앞에 $A^+$ 곱하기
- 응용 2 : 선형회귀분석에서 데이터가 변수 개수보다 많거나 같을 때 사용 (행이 더 많기 때문에 방정식을 푸는 것은 불가능 → L2 norm 을 최소하하는 웨이트 찾기 가능), 양 변 앞에 $X^T$ 곱함
    - `sklearn` 의 `LinearRegression` 과 같은 결과를 가져올 수 있음

    ```python
    # Scikit Learn 을 활용한 회귀분석
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_test = model.predict(x_test)

    # Moore-Penrose 역행렬
    X_ = np.array([np.append(x, [1]) for x in X]) # y 절편 (1) (intercept) 항을 직접 추가해야함
    beta = np.linalg.pinv(X_) @ y # @ 는 행렬곱
    y_test = np.append(x, [1]) @ beta
    ```

⇒ 딥러닝을 이해하려면 행렬 연산에 대한 완벽한 이해가 필요! 꼭 복습할 것

<br>

<hr>

<br>


# 피어 세션

## 주말 회고

- 히스 : 윤스테이, 알고 공부, 광안리 산책
- 후미 : 다른 프로젝트 하고 롤챔스 봄
- 펭귄 : 거의 잠만 잠, 유튜브 봄
- 샐리 : 잠을 거의 못잠, 블로그 만들었는데 오류나서 고침
- 엠제이 : 잠 거의 못잠, 술도 먹고 논문 스터디 하고 넥슨 데이터 분석 하는데 잘 안되서  시간 씀
- 원딜 : 잠 자고 더지니어스 장동민 재생목록 다봄
- 서폿 : 실화탐사대 봄

## 강의 관련

- a 어레이에 대해서 a[1] 하면 [1, 2, 3] 이런식으로 나오고 a[1:3] 하면 [[1, 2, 3]] 이 나옴
- np.newaxis 쓰면 차원이 늘어나는데, 대신 None 써도 됨
- axis 로 연산할 때 axis 가 사라진다고 생각하면 편함
- 파이썬에는 기본 array가 없는가? 
: 파이선의 list 가 다른 언어의 배열 역할을 함

<br>

<hr>

<br>

# Today I Felt

## 손으로 따라하기

강의를 보면 다양한 예제가 나오는데 하나하나 따라하다보면 시간이 오래 걸린다. 하지만 오늘 피어 세션에서 노트 정리한 내용들을 같이 보니 하나하나 따라하는 조원들이 많았고 시간이 걸려도 익숙해지려면 직접 손으로 따라하는게 맞구나라는 생각이 들었다. 능숙하게 다를 수 있는 부분은 스킵하되 익숙지 않은 부분은 꼭 따라해야지.
