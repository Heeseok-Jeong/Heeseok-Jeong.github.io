---
layout : post
title : Ustage Day 3
subtitle : 파이썬 자료구조와 파이써닉 코딩
tags : [BoostCamp AI Tech]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

- [Python Data Structure](#python-data-structure)
- [Pythonic Code](#pythonic-code)
- [피어 세션](#피어-세션)
- [느낀점](#느낀점)

<br>

<hr>

<br>

# Python Data Structure

## Stack

- 나중에 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조 (LIFO, Last In First Out)
- 데이터 입력을 push, 출력을 pop 이라 함
- list 로 스택 가능
    - list.append()
    - list.pop() : 특이한게 리턴 (나오는 값) 이 있으면서도 리스트가 변함

## Queue

- 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조 (FIFO, First In First Out)
- 스택과 반대되는 개념
- 리스트를 사용하여 큐 구조 가능
    - list.append()
    - list.pop(0) : 처음 원소 빠짐, O(N) 으로 비효율적
- collections.deque 로 queue 사용하도록 하자
    - deque.append()
    - deque.popleft()

## Tuple

- 값의 변경이 불가능한 리스트
- 선언시 [] 가 아닌 () 를 사용
- 리스트의 연산, 인덱싱, 슬라이싱 등을 동일하게 사용
- 왜 쓸까?
    - 프로그램을 작동하는 동안 변경되지 않는 데이터 저장
    - 함수의 리턴값 등 사용자의 실수에 의한 에러를 사전에 방지

## Set

- 값을 순서없이 저장, 중복 허용하지 않는 자료형
- set([1, 2]) 로 선언
- 내장 함수
    - add : 한 원소 추가
    - remove : 한 원소 삭제
    - update : 한 번에 원소들 추가
    - s1.union(s2) or s1 \(shift) s2 : 합집합
    - s1.intersection(s2) or s1 & s2 : 교집합
    - s1.difference(s2) or s1 - s2 : 차집합

## Dict

- 데이터를 키와 밸류로 함께 저장
- 키는 데이터 고유 값, Identifier !
- dict() or {} 로 선언
- {key1 : value1, key2 : [value2, value3]}
- 내장 함수
    - items : tuple 형태로 키 밸류가 나옴, 반복문에서 언패킹 k, v 해서 많이 사용
    - keys : key 값들만 묶여서 나옴
    - values : values 값들만 묶여서 나옴

### 실습 : Command Analyzer

- 커맨드 명령어 분석

## Collections

- List, Tuple, Dict 에 대한 파이썬 빌트인 확장 자료 구조 (모듈)
- 편의성, 실행 효율 제공
- 목록
    - deque
    - Counter
    - OrderedDict
    - defaultdict
    - namedtuple

### deque

- 스택과 큐를 동시에 지원하는 모듈
- 리스트에 비해 효율적인 자료 저장 방식 적용
- 링크드 리스트 특성 지원
- 기존 리스트 함수 모두 지원

### defaultdict

- 일반 딕트는 키값이 생성되지 않으면 에러가 나는데, 디폴트 딕트는 키값 없으면 디폴트 값을 넣어줌

    ```python
    from collections import defaultdict

    d = defaultdict(lambda : 0)
    d["aa"] # 0
    ```

- 텍스트에서 데이터 마이닝할 때 딕트나 디폴트 딕트 이용하면 편리함

### Counter

- sequence type 의 데이터 엘리멘트 개수를 세주는 모듈
- set 의 연산 지원 (합, 교, 차집합)

### namedtuple

- tuple 형태로 데이터 구조체를 저장하는 방법
- 저장되는 데이터의 variable 을 사전에 지정해서 저장

    ```python
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(x=11, y=22)
    p[0], [1] # (11, 22)
    p # (Point(x=11, y=22)
    p.x + p.y # 33

    ```

<br> 

<hr> 

<br>

# Pythonic Code

- 파이썬 스타일의 코딩법
- 파이썬 특유의 문법을 활용하여 효율적으로 코드를 표현함
- 이제는 많은 언어들이 서로의 장점을 채용, 고급 코드 작성할 때 더 많이 필요해짐
- 컴퓨터에게 시간이 더 들어도 사람의 시간은 덜 들어서 좋음
- 왜 쓰는가?
    - 다른 사람 코드에 대한 이해도 (많은 개발자들이 파이썬 스타일로 코딩함)
    - 효율성 (속도도 빠르고 익숙해지면 코드도 짧아짐)
    - 간지 (잘 짜는 거처럼 보임)

## split & join

- split
    - string type 의 값을 기준값으로 나눠서 list 형태로 변환
- join
    - 스트링 리스트를 기준값으로 붙여서 스트링으로 변환

## list comprehension

- 기존 리스트를 사용하여 간단히 다른 리스트를 만드는 기법 (포괄적인 리스트, 포함되는 리스트라는 의미)
- 파이썬에서 가장 많이 사용되는 기법 중 하나
- 일반적으로 for + append 보다 속도가 빠름
- Nested For loop
    - a = [i*j for i in range(1, 3) for j in range(10, 30, 10)] # [10, 20, 20, 40]
- Filter (조건)
    - a = [i for i in range(10) if i % 2 == 0] # 0~9 중 짝수만 a 에 넣기
- 2차원 for loop
    - a = [[i+j for i in range(5) if i % 2 == 0] for j in range(5) if i % 2 == 1]

### pprint

- 이쁘게 출력

```python
import pprint
pprint.pprint(a)
```

## enumerate & zip

- enumerate : 리스트의 엘리먼트를 추출할 때 번호를 붙여서 추출

    ```python
    for i, v in enumerate("ABC")
    	print(f"{v} : {i}") 
    # A : 0
    # B : 1
    # C : 1
    ```

- zip : 두 개의 리스트 값을 병렬적으로 추출 (집은 제너레이터라 리스트로 감싸줘야함)

    ```python
    a = ["a", "b", "c"]
    b = ["A", "B", "C"]

    [c for c in zip(a, b)]
    # [("a", "A"), ("b", "B"), ("c", "C")]
    ```

## lambda & map & reduce

- lambda : 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수 (사실 파이썬3 부터 권장안함, 하지만 많이쓰임)
    - f = lambda x, y: x + y
    - 어려운 문법, 테스트도 어려움, 문서화 docstring 지원 미비 등 문제가 있다.
- map : 두 개 이상의 리스트에도 적용 가능, if filter 도 사용가능 (최근에는 리스트 컴프리헨션으로 쓰는게 낫다 말하지만 많이 씀)
    - list(map(함수, 리스트1, 리스트2..) : 원소들에 대해 함수 적용
- reduce : 맵과 달리 리스트에 똑같은 함수를 적용해서 통합 (직관성이 떨어져서 평소엔 많이 안쓰임, 대용량 데이터 다룰 때 씀)

    ```python
    from functools import reduce
    print(reduce(lambda x, y: x+ y, [1, 2, 3, 4, 5])) # 15
    ```

## Iterable Object

- Sequence형 자료형에서 데이터를 순서대로 추출하는 object
- 내부적으로 iter 와 next 함수로 구현됨
    - iter 는 메모리 구조를 가지고 옴
    - next는 메모리를 참고해서 값을 가져옴, 그리고 메모리 다음으로 옮김
- 리스트 객체는 한 뭉치의 메모리가 있고 각각 주소에서 자기 값에 해당하는 주소를 바라보는 형태

## Generator

- 효율적인 메모리 사용, 일반적으로 제너레이터 권장

```python
import sys
def general_list(v):
	result = []
	for i in range(v):
		result.append(i)
	return result

a = general_list(50)
sys.getsizeof(a) # 520 (Bytes)

def generator_list(v):
	result = []
	for i in range(value):
		yield i # 호출할 때 데이터를 출력해줌

b = generator_list(50) # 평소엔 메모리 주소값만 가지고 있다가 호출될 때마다 던져줌, 효율적
sys.getsizeof(b) # 112 (Bytes)
```

- generator comprehension (= generator expression)
    - 리스트 컴프리헨션과 유사한 형태로 제너레이터 형태의 리스트 생성
    - [ ] 대신 ( ) 사용

    ```python
    gen_ex = (n*n for n in range(500))
    type(gen_ex) # generator, 주소만 있고 메모리 할당은 안함
    list(gen_ex) # 메모리 생성
    ```

- 왜 쓸까? 일반적인 iterator 보다 훨씬 적은 메모리 사용
- 언제 쓸까?
    - 리스트 타입의 데이터를 반환해주는 함수는 generator로 만들어라 (yield)
        - 읽기 쉬운 장점, 중간 과정에서 loop 이 중단될 수 있을 때
    - 큰 데이터 처리할때 generator expression 고려할 것
    - 파일 데이터 처리할 때도 generator 쓰자

## Function passing arguments

- Keyword arguments
    - 함수에 입력되는 파라미터의 변수명을 사용, 아규먼트를 넘김
    - func(x = 10)
- Default arguments
    - 파라미터의 기본 값 사용, 넣어주면 넣은 값 사용
    - def func(x = 1):
- Variable-length asterisk (가변길이 에스터리스크*)
    - 함수의 파라미터가 정해지지 않음 (다항 방정식 등)

        → Asterisk(*) 를 사용하여 개수가 정해지지 않은 변수를 함수의 파라미터로 사용

    - 입력된 값은 tuple type 으로 사용 가능

    ```python
    def asterist_test(a, b, *args):
    	return a+b+sum(args)

    print(asterisk_test(1, 2, 3, 4, 5) # 15K
    ```

- Keyword variable-length (키워드 가변인자)
    - 파라미터 이름을 따로 지정하지 않고 입력
    - Astserisk 2개 사용
    - 입력된 값은 dict_type으로 사용
    - 가변인자는 오직 한 개만 기존 가변인자 다음에 사용

    ```python
    def kwargs_test(**kwargs):
    	print(kwargs)
    	print(type(kwargs))

    kwargs_test(first=3, second=4, third=5)
    # {'first' : 3, 'second' : 4, 'third' : 5}
    # <class 'dict'>

    def kwargs_test_2(one, two, *args, **kwargs):
    	print(one + two + sum(args))

    kwargs_test_2(10, 30, 3, 5, 6, 7, first = 3, second = 4, third = 5)
    # one = 10, two = 30, args = (3, 5, 6, 7), kwargs = {'first' : 3, 'second' : 4, 'third' : 5}
    # 가변인자 순서 지켜줘야 함
    ```

## asterisk 의 또다른 기능 - unpacking a container

- tuple, dict 등 자료형에 들어가 있는 값을 언패킹
- 함수의 입력값, zip 등에 유용하게 사용가능

    ```python
    def ast1(a, *args):
    	print(a, args)
    	print(type(args))

    test = (2, 3, 4, 5)
    ast1(1, *test) # 입력할 때 별표치면 풀려서(언패킹) 들어감
    # 1 (2, 3, 4, 5)
    <class 'tuple'>

    def ast2(a, args):

    ```

- 함수 입력할 때 변수에 * 를 붙이면 풀려서 들어간다.
- * 두 개 쓰면 dict key = value 됨

<br>

<hr>

<br>

# 피어 세션

1. Generator 의 yield 는 어떻게 메모리를 할당하고 어떤 방식으로 동작하는가?
: 아직 답을 못찾아서 따로 블로깅 예정!
2. 이론에서는 메모리 공간에 관해 배우는데 일상적으로 메모리 (공간) 를 고려하는 일이 많지 않아보이는데 어떤 상황에서 고려해야할까
: IoT 나 임베디드 컴퓨팅 같이 하드웨어가 작을 때!

<br>

<hr>

<br>

# 느낀점

## 파이써닉한 코딩

리스트 컴프리헨션부터 애스터리스크 인자까지 다양한 파이썬만의 코딩 스타일을 배울 수 있었다. 기존에 딥러닝 모델 적용과 코딩테스트를 하면서 리스트 컴프리헨션 정도는 눈치껏 배웠었다. 하지만 zip 이나 map 이 파이써닉 한 것인지 몰랐고, 애스터리스크 문법은 처음 배우는 것이라 유익했다. 

## 스타일 변화

기존에 하던 습관을 바꾸기란 참 쉽지 않다. 조원들을 보면서 마크다운을 더 잘 써가며 블로깅 해야할 필요성과 모르는 부분에 있어서 대충 알지 않고 끝까지 파고들어 알아내는 습관을 기르고 싶다는 생각이 들었다. 또 파이써닉 코딩 스타일을 보면서 더 멋지게 코딩해야함을 배웠다. 사실 기존 습관이 익숙하다는 것은 귀찮음에 기반한 핑계임을 안다. 습관 바꾸기 쉽지 않겠지만 더 발전하기 위해 변화를 주리라.
