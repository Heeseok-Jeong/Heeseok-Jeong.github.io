---
layout : post
title : Python Mock for unittest
subtitle : 유닛테스트 외부환경 대체제
tags : [Python]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

<br>
- [Mock 이란?](#mock-이란)
- [patch 함수](#patch-함수)
- [결론](#결론)


<br>
<hr>
<br>

# Mock 이란?
<br>

`mock` 의 원래 뜻은 모조품, 모조 객체이다. 유닛 테스트를 수행할 때 단순히 맞고 틀린지 등을 비교하는 `assert` 계열의 함수만으로 테스트가 충분하지 않을 때가 있다. 테스트를 위해 외부 API, 네트워킹이나 db 사용 등의 환경이 필요할 때, `mock` 이 이들을 대신하여 테스트 환경을 지원한다. 즉, `mock` 은 테스트에 필요한 외부 환경에 의존하지 않고 독립적으로 테스트를 수행할 수 있게 해주는 기특한 툴이다.   
<br>

`mock` 객체는 `return_value` 와 `side_effect` 옵션을 가진다. 객체를 생성할 때 지정해줘도 되고 객체 생성 후에 직접적으로 옵션을 설정해줘도 무방하다.   
<br>

## Mock 객체 기본 생성
<br>

`return_value` 옵션은 생성된 mock 객체를 호출했을 때 값을 설정할 수 있다.
```python
  from unittest.mock import Mock

  my_mock = Mock(return_value = "I'm mock.")
  print(my_mock)
  # I'm mock.
  my_mock.return_value = "Neck slice")
  print(my_mock)
  # Neck slice
```

<br>

`side_effect` 옵션은 mock 객체를 호출할 때 지정한 익셉션을 호출 할 수도 있고, 아니면 호출할 때마다 `iterator` 의 `next` 함수처럼 값을 하나씩 호출하도록 설정할 수도 있다.
```python
  from unittest.mock import Mock

  # 익셉션 호출
  my_mock = Mock(side_effect = Exception("Mock Error"))
  print(my_mock)
  # Exception: Mock Error 익셉션 발생

  # 값 한개씩 호출
  my_mock = Mock(side_effect = [1, 2, 3])
  print(my_mock)
  # 1
  print(my_mock)
  # 2
  print(my_mock)
  # 3
  print(my_mock)
  # StopIteration 익셉션 호출
```
<br>
<br>

## Mock 객체 내장함수
<br>

|함수이름|기능|
|------|---|
|call_args|마지막으로 설정한 arguments 리턴|
|call_count|몇 번 호출되었는지 리턴|
|assert_called|한 번도 호출되지 않았으면 익셉션 발생, 호출된 적 있으면 통과|
|assert_called_once|딱 한 번만 호출 되었으면 통과|
|assert_not_called|한 번도 호출되지 않았으면 통과|


<br>
<hr>
<br>

# patch 함수
<br>

`mock` 객체는 보통 위에 적힌 기본 생성 방법으로 생성하지 않고, 내장함수 표에서 언급하지 않은 내장함수 `patch` 를 사용하여 외부 API 에 독립적인 객체를 생성해낸다. `patch` 로 객체를 만드는 방법에는 decorator 를 사용하거나 with 를 사용하는 방법이 있다.  
<br>

어떤 모듈이 사용자로부터 소문자 영어 알파벳을 입력받아 대문자로 프린트해준다고 가정하자. 그리고 사용자로부터 0을 입력받으면 종료 멘트와 함께 프로그램이 종료된다. 이 모듈을 테스트하는 코드에서 여러 인풋값 (상황) 을 설정해두고 그 결과를 플로우대로 검사하고 싶을 때 `mock` 을 사용하면 유용하다. 

> test_module.py

```python
  import unittest
  from unittest.mock import patch
  import module

  class TestModule(unittest.TestCase):
    def test_main(self):
      # 입력은 반드시 영어 소문자나 0만 들어옴을 보장
      input_list = ['a', 'd', 0]

      # with 를 사용한 mock 객체 생성
      with patch('builtins.input', side_effect=input_list):
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
          module.main()
          console = fakeOutput.getvalue().strip().split("\n")
          self.assertIn("A", console[0].upper())
          self.assertIn("D", console[1].upper())
          self.assertIn("BYE", console[2].upper())

```

<br>
<hr>
<br>

# 결론
<br>

> `mock` 은 유닛 테스트에서 외부 환경에 대한 의존성을 없애고 독립적으로 테스트를 수행할 수 있게 외부 환경 객체 역할을 대신해준다. `mock` 은 `patch` 를 이용해 decorator 또는 with 구문으로 생성하여 사용한다.
