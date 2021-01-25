---
layout : post
title : Python 동시 비교의 함정
subtitle : print(4 != 0 not in [0, 1, 2, 3]) => False
tags : [Python]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

<br>

- [편리한 동시 비교](#편리한-동시-비교)
- [동시 비교의 함정](#동시-비교의-함정)
- [결론](#결론)

<br>
<hr>
<br>

# 편리한 동시 비교
(* 정확한 용어를 몰라 동시 비교라고 썼습니다)

<br>

C 와 같은 많은 언어와 다르게 파이썬은 동시 비교를 지원한다. 예를 들어 `a < b and b < c` 연산을 간단하게  
`a < b < c` 이렇게 작성할 수 있는 것이다.  
<br>   

내부적으로 동시비교 연산은 `피연산자1 연산자1 피연산자2 연산자2 피연산자3` 가 있으면,   
`피연산자1 연산자1 피연산자2 and 피연산자2 연산자2 피연산자3` 으로 변환된다. 
<br>

```python
if 1 < 2 < 3:
    print("We are")

if 1 < 2 and 2 < 3:
    print("Same")
```


<br>
<hr>
<br>

# 동시 비교의 함정

<br>

이렇게 편리해보이기만한 동시 비교는 쉽게 프로그래머로 하여금 함정에 빠지게 한다. 

1. 4 != 0 not in [1, 2, 3]
    => True
2. (4 != 0) not in [1, 2, 3]
    => False
3. 4 != 0 not in [0, 1, 2, 3]
    => False
4. (4 != 0) not in [0, 1, 2, 3]
    => False

<br>

이 문제를 쉽게 맞췄다면 동시 비교 개념을 잘 이해하고 있는 것이다. 하지만 잘 모르겠다면 다시 기본 원리를 생각해보자.  
1 번과 3 번은 `피연산자1 연산자1 피연산자2 연산자2 피연산자3` 꼴이므로 `피연산자1 연산자1 피연산자2 and 피연산자2 연산자2 피연산자3`
식으로 치환된다.  
반면, 2 번과 4 번은 괄호로 인해 괄호를 먼저 수행하고 나면 `피연산자1 연산자1 피연산자2` 의 형태가 된다. 

<br>
<hr>
<br>

# 결론

<br>

> `피연산자1 연산자1 피연산자2 연산자2 피연산자3` 연산은 `피연산자1 연산자1 피연산자2 and 피연산자2 연산자2 피연산자3'
연산이다.

<br>
<hr>
<br>

# 참조
<br>

- [6.10 Comparisons](https://docs.python.org/3/reference/expressions.html#comparisons)

<br>
