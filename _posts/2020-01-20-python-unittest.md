---
layout : post
title : Python Unittest 
subtitle : 단위 테스트
tags : [Python]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

<br>

- [단위 테스트란?](#단위-테스트란?)
- [파이썬에서는 어떻게?](#파이썬에서는-어떻게?)
- [결론](#결론)

<br>
<hr>
<br>

# 단위 (Unit) 테스트란?
<br>

일반적으로 테스트는 어떤 대상이 목적에 맞게 행동하는지를 검사하는 것이다. 소프트웨어에서 테스트도 마찬가지이다.  
<br>
  
소프트웨어가 여러 개의 동작을 하는 함수들로 구성돼 있을 때, 한 번에 모든 기능이 잘 동작하는지 살펴보는 것은 좋지 않다. 실패할 경우 어떤 부분에서 기능이 동작하지 않는지 알기 어렵기 때문이다. 따라서 단위 (Unit) 별로 기능을 테스트 해야한다.  
<br>
  
보통 유닛은 하나의 동작을 하도록 설계한다. 그러므로 단위 테스트는 하나의 동작이 잘 수행하는지 하나씩 살펴보는 작업이다.

<br>
<hr>
<br>

# 파이썬에서는 어떻게?
<br>

파이썬은 유닛 테스트를 위한 각종 기능을 사용할 수 있는 `unittest` 모듈을 지원한다.
<br>  
  
먼저 `unittest.TestCase` 를 상속한 테스트 케이스를 작성하고 그 밑에 픽스쳐부터 테스트까지 함수들을 정의한다.  
<br>
  
픽스쳐는 테스트 전, 후에 필요한 과정을 수행하는 동작을 말하며 이미 지정된 네이밍을 이용하여 이 단계를 수행하면 된다. 테스트 전에 실시해야할 사항을 수행하는 함수는 `setUp` 이고, 테스트 후에 실시해야할 사항을 수행하는 함수는 `tearDown` 이다.  
<br>

이제 테스트 수행 단계만 남았다. 테스트를 위해서는 테스트케이스 밑에 `test_` 로 시작하는 함수를 정의한다. 그리고 확인하고픈 유닛(메서드)을 `assert` 계열 함수들로 검사해주면 된다. 그 중 하나인 `assertEqual` 는 첫 번째 인자로 유닛의 결과를 넣고 두 번째 인자로 정답을 넣어주면, 둘이 같을 경우 통과되고 그렇지 않으면 틀렸음을 알려준다.
<br>
<br>

> [assert 함수 더 보기](https://docs.python.org/ko/3/library/unittest.html#assert-methods)

<br>
<br>

### 예시
  
`arith.py` 라는 어떤 수행에 필요한 기능들을 담은 파일이 있다면 이 파일을 모듈로 불러와 안에 있는 유닛 메서드들을 검사하는 `test_arith.py` 를 만들어 테스팅을 진행하면 된다.   
  
> arith.py

```python
def add(a, b):
    return a+b

def substract(a, b):
    return a-b
```

> test_arith.py

```python
import unittest
import arith

class TestArith(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_add(self):
        x, y = 1, 2
        result = arith.add(x, y)
        self.assertEqual(result, 3)

    def test_subtract(self):
        x, y = 1, 2
        result = arith.subtract(x, y)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
```
<br>

### 결과

- `test_add` 테스트 통과
- `test_subtract` 테스트 실패, 결과 -1 과 예측값 1 이 다르므로 

<br>
<br>


## 테스트 호출의 다른 방법

위에 테스트 파일을 보면 `if __name__ == "__main__":` 로 테스트를 호출한 것을 볼 수 있는데 이 방법 대신에 터미널에서 `python -m unittest test_arith.py` 로 테스팅을 수행할 수 있다.  

<br>
<hr>
<br>

# 결론
<br>

> 하나의 기능을 지닌 유닛을 테스트하는 `unittest` 모듈을 사용하여 유닛 테스트를 진행할 수 있다. 테스트 케이스를 정의하고 그 밑에 테스트할 함수들을 만들어 테스트를 수행한다.

<br>
<hr>
<br>

# 참조
<br>

[50. unittest - 단위테스트](https://wikidocs.net/16107)
