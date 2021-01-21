---
layout : post
title : Python Generator 
subtitle : Generator, Iterator, Iterable
tags : [Python]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차
<br>

- [효율적인 메모리 공간](#효율적인-메모리-공간)
- [결론](#결론)

<br>
<hr>
<br>

# 효율적인 메모리 공간

<br>

리스트에 1 부터 100 까지 값을 넣으면 항상 최대 크기로 메모리를 잡아먹기 때문에 비효율적이다. 작은 양의 데이터를 다룰 때는 큰 문제가 되지 않지만 많은 양의 데이터를 다룰 때 이는 문제가 된다. 그렇다면 효율적으로 데이터를 메모리에 할당하고 사용할 수 있는 방법은 없을까?  

<br>

## 정답은 Generator

제너레이터는 `Iterator` 를 편하게 생성해주는 함수이다. `Iterator` 는 메모리에 모든 데이터를 담고 있지 않고 호출될 때만 하나씩 데이터를 생성해주기 때문에 효율적으로 메모리를 사용할 수 있다.  

<br>

제너레이터로 `Iterator` 를 만드는 데에는 두가지 방법이 있다.  
<br>

(1) `yield` 를 사용하는 제너레이터 함수
: 함수 안에서 `yield` 로 값을 넘겨주면 메모리만 기억해놓고 실제 사용할 때 값을 할당한다.

<br>

```python
def generate_range(v):
    for i in range(v):
        yield i

v = 3
for x in generate_range(v):
    print(x)
# 0
# 1
# 2

```

<br>

(2) generator comprehension (= expression)
: 리스트 컴프리헨션과 비슷하지만 [ ] 가 아닌 ( ) 로 표현하면 제너레이터가 된다.  
-> `(i for i in range(v))`

<br>

```python
for x in (i for i in range(v)):
    print(x)
# 0
# 1
# 2

```

<br>

## Iterator 와 Iterable
`Iterator` 와 `Iterable` 은 이름은 비슷하지만 다르다.   
<br>

먼저 `Iterable` 은 seqence 형태로 반복적인 원소를 지닌 데이터 구조를 말한다. 리스트, 셋, 딕트 등이 포함된다.   
<br>

`Iterator` 는 `Iterable` 객체를 받아서 next 함수를 통해 원소를 하나씩 살펴본다. 단, 모든 원소를 다 돌면 더이상 볼 메모리가 없기 때문에 `StopIteration` exception 을 호출하고, 만약 다시 이미 봤던 원소 조회를 하고 싶다면 새로운 `Iterator` 객체를 생성해야한다. 왜냐하면 메모리를 하나씩 참조하면서 이미 본 것은 삭제해버리기 때문이다.  

<br>
<hr>
<br>

# 결론

<br>

> `Generator` 는 `Iterator` 를 `yield` 함수나 `generator comprehension` 을 통해 생성해주는 함수이고 `Iterator` 는 사용될 때에만 메모리를 하나씩 참조하여 일반적인 자료 구조를 사용하는 것보다 효율적으로 메모리를 관리할 수 있다.

