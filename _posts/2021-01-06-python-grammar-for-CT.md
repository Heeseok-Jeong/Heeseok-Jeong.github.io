---
layout : post
title : Python Grammar for Coding Test
subtitle : 동빈북 부록 A + My tips
tags : [Problem Solving, Python]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목록
- [소수](#소수)
- [2차원 배열 만들기](#2차원-배열-만들기)
- [리스트 함수](#리스트-함수)
- [문자열](#문자열)
- [입력](#입력)
- [내장함수](#내장함수)
- [itertools](#itertools)
- [heapq](#heapq)
- [bisect](#bisect)
- [collections](#collections)
- [math](#math)
- [dict 에서 value 를 키로, key 를 밸류로](#dict-에서-value-를-키로-key-를-밸류로)

<br>
<hr>
<br>

### 소수


0.3 + 0.6 = 0.8999999. 가 나옴

⇒ round(0.3 + 0.6, 1) = 0.9

<br>


### 2차원 배열 만들기

n : row, m : col

arr = [[0] * m for _ in range(n)]

or

arr = [[0 for _ in range(m)] for _ in range(n)]

<br>

### 리스트 함수

- append만 O(1), insert와 remove는 O(N)
- arr.sort() : 오름차순, arr.sort(reverse = True) : 내림차순
- arr.reverse() 거꾸로 뒤집어줌
- arr.rotate(1) 오른쪽으로 밂

<br>

### 문자열

슬라이싱 [2:4] 잘 사용할 것

<br>

### 입력

- 빠르게

    ```python
    import sys
    input = sys.readline().rstrip()
    a = input()
    ```

- 한 줄에 여러개 입력 들어올 때 (1 2 3)
arr = list(map(int, input().split()))
a, b, c = map(int, input().split())

<br>

### 내장함수

- sorted, 람다식으로 키값 설정해서 정렬기준 만드는 예제, 첫 원소 기준 오름차순, 같은경우 두번째 원소로 내림차순
arr = sorted(arr, lambda key = x : (x[0], -x[1]))

<br>

### itertools

- list(permutations(arr, 2)) : nP2 ⇒ 순열들 튜플 형태로 리스트에 저장
- list(combinations(arr, 2)) : nC2 ⇒ 조합들 튜플 형태로 리스트에 저장

<br>

### heapq

- PriorityQueue보다 빨라서 대신 사용 권장

```python
import heapq
#삽입
heapq.heappush(arr, x)
...
#뺄 때, 오름차순으로 빼줌
a = heapq.heappop(arr)

#내림차순
haepq.heappush(arr, -x)
a = -heapq.heappop(arr)
```

<br>

### bisect

- 이진탐색 라이브러리, bisect_left()와 bisect_right()함수 O(logN)으로 정렬된 배열에서 특정 원소 위치 찾을 때 효과적
- arr = [1, 2, 4, 4, 8] 배열에서 bisect_left(arr, 4) 하면 4의 첫 위치 2 리턴, bisect_right(arr, 4)하면 4의  마지막 위치보다 뒤인 4를 리턴함.
- 이를 활용하여 정렬된 배열에서 4의 개수를 알고 싶다면, bisect_right(arr, 4) - bisect_left(arr, 4) 를 하면 4-2 = 2가 나온다.
- 다른 활용법으로 정렬된 배열에서 [2, 4] 사이에 있는 원소 개수를 알고 싶다면, bisect_right(arr, 4) - bisect_left(arr, 2)를 하면 4-1 = 3이 나온다.

<br>

### collections

deque

- 큐를 구현할 때, 리스트가 아닌 디큐를 사용함, 리스트는 오른쪽에 넣고 빼는 작업은 O(1)이지만 왼쪽에선 O(N)이기 때문. deque는 appendleft()와 popleft() 제공

Counter

- Counter(arr).most_commont() 하면 원소들이 튜플 형태로 리스트에 등장 횟수가 많은 원소부터 기록됨
- dict(Counter(arr)) 하면, 딕셔너리에 원소가 키가되고 횟수가 밸류가 되어 저장됨

<br>

### math

- math.factorial(5) : 5! 계산
- math.sqrt(5) : 루트5 = 2.6457... 계산
- math.gcd(14, 21) : 최소공배수 계산 ⇒ 7

*자신만의 알고리즘 노트 만들기

- 좀 어려운 문제 풀면 참고할 수 있게 내용과 어떻게 풀었는지 기록
- 2차원 리스트를 회전하는 함수를 만들었다 치면 그 함수 코드와 사용법까지 기록할 것
- 깃허브 같은 곳에 저장하는거 추천

<br>

### dict 에서 value 를 키로, key 를 밸류로

```python
reversed_dict = dict(map(reversed, original_dict.items()))
```

### json parsing
```python
>>> import jmespath
>>> persons = {
...   "persons": [
...     { "name": "erik", "age": 38 },
...     { "name": "john", "age": 45 },
...     { "name": "rob", "age": 14 }
...   ]
... }
>>> jmespath.search('persons[*].age', persons)
[38, 45, 14]
```
