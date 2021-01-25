---
layout : post
title : Ustage Day 4
subtitle : 파이썬 OOP 와 모듈|패키지|프로젝트
tags : [BoostCamp AI Tech]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

- [Python Object Oriented Programming](#python-object-oriented-programming)
- [Module and Project](#module-and-project)
- [피어 세션](#피어-세션)
- [Today I Felt](#today-i-felt)

<br>
<hr>
<br>

# Python Object Oriented Programming

<br>

## 만들어 놓은 코드를 재사용하고 싶다면?

- 객체를 가져와서 사용한다.
- 객체는 클래스를 통해 구현된다.

## 객체지향 프로그래밍 개요

- 객체 : 실생활의 물체 → 속성(어트리뷰트) 과 행동(메서드) 를 가짐
- 파이썬은 객체 지향 프로그램 언어
- 변수 타입, 변수명 등 모두 객체
- OOP 는 설계도에 해당하는 **클래스**와 실제 구현체인 **인스턴스(객체)** 로 나눔

## Objects in Python

- class 선언
    - object 는 python3 에서 자동 상속, class 예약어 - class 이름 - 상속받는 객체명 순으로 선언
    - 변수 이름은 소문자에 언더바를 이용해 (snake_case) 만든 반면, 클래스 이름은 띄어쓰기 부분에 대문자를 사용 (CamelCase)
    - class SoccerPlayer(object):
- Attribute 추가
    - __init__(객체 초기화 예약 함수), self 를 통해 선언.
    - 파이썬에서 언더바 두개 __ (맨글링) 의 의미 : 특수한 예약함수나 변수 그리고 함수명 변경으로 사용
- method 구현
    - 기존 함수처럼 추가하되, 반드시 self 를 추가해야 class 함수로 인정됨
    - init, str 등 매직메서드 (기존에 사용이 정의된 함수) 들 존재함
- objects(instance) 사용
    - hm = SoccerPlayer("SHM", "FW", 7)
- self : 생성된 인스턴스 자신

## OOP 특성

→ 실제 세상을 모델링 했기 때문에 필요한 것들이 있음

### Inheritance (상속)

- 부모 클래스로부터 속성과 메서드를 물려받은 자식 클래스를 생성하는 것
- super() : 부모 클래스의 값을 가져올 수 있음

### Polymorphism (다형성)

- 같은 이름의 메소드를 내부 로직을 다르게 작성
- Dynamic Typing 특성으로 인해 파이썬에서는 같은 부모클래스의 상속에서 주로 발생 (오버라이딩)

### Visibility (가시성)

- 객체의 정보를 볼 수 있는 레벨을 조절하는 것, 누구나 객체 안에 모든 변수를 보는 것은 보안에 위험
    1. 객체를 사용하는 사용자가 임의로 정보 수정의 위험
    2. 필요 없는 정보에는 접근할 필요가 없음
    3. 제품으로 판다면 소스코드 보호해야 함
- Encapsulation (캡슐화 or 은닉 (Information Hiding))
    - 클래스 간 간섭, 정보공유 최소화
    - 심판 클래스가 축구선수 클래스 가족 정보 알 필요 없음
    - 캡슐 던지듯, 인터페이스만 알아서 써야함
- private 변수 선언 : init 에서 self.__items(변수명) 하면 프라이빗 돼서 외부에서 직접 접근 불가
    - 그럼 프라이빗 변수에 접근하고 싶을 때는?
        - @property (property decorator) 를 사용!

        ```python
        class Inventory(object):
        	def __init__(self):
        		self.__items = []

        	@property
        	def items(self):
        		return self.__items
        ```

## decorate

- @ 붙은애들 : decorator

### First-class objects

- 일등함수 또는 일급 객체
- 변수나 데이터 구조에 할당이 가능한 객체
- 함수를 **파라미터로 전달**이 가능 + **리턴** 값으로 사용

→ 파이썬의 함수는 일급함수

### Inner function

- 함수 내에 또 다른 함수가 존재

    ```python
    def print_msg(msg):
    	def printer():
    		print(msg)
    	printer()

    print_msg("Hi")

    # closures
    def print_msg(msg):
    	def printer():
    		print(msg)
    	return printer

    another = print_msg("Hello")
    another()
    ```

- closures : inner function 을 return 값으로 반환 (js 에서 매우 많이 사용)
    - 장점 : 한 함수로 태그 펑션처럼 펑션을 목적에 따라서 다양한 함수처럼 사용 가능

### decorator function

- 복잡한 클로져 함수를 간단하게 만들어줌

    ```python
    # Hello 가 msg 로 들어감 -> printer 가 func 로 들어감 -> 
    # 별표 30개 찍고 func (printer) 작동해서 Hello 찍고 다시 별표 30개 찍음
    def star(func):
    	def inner(*args, **kwargs):
    		print("*" * 30)
    		func(*args, **kwargs)
    		print("*" * 30)
    	return inner

    @star
    def printer(msg):
    	print(msg)
    printer("Hello")

    # ***...***
    # Hello
    # ***...***

    # 변형
    def star(func):
    	def inner(*args, **kwargs):
    		print(args[1] * 30)
    		func(*args, **kwargs)
    		print(args[1] * 30)
    	return inner

    @star
    def printer(msg, mask):
    	print(msg)
    printer("Hello", 0)

    # 000...000
    # Hello
    # 000...000
    ```

- decorator 에도 아규먼트 넣을 수 있음. 그러면 inner 전에 wrapper 함수 만들어야함.

Q1. Inner function, closures, decorator function 왜 쓰는지 잘 모르겠다. 사례를 더 찾아보면 좋을듯

<br>
<hr>
<br>

# Module and Project

<br>

- 파이썬은 대부분의 라이브러리가 이미 다른 사용자에 의해 구현되어 있음

## 모듈

- 프로그램에서 사용하는 작은 조각들
- 모듈들을 모아서 하나의 큰 프로그램 생성
- 파이썬의 모듈 = .py 파일
- import 문을 사용해서 module 호출
- __pycache__ 가 생김 → 메모리 로딩을 더 빠르게 하기위해 컴파일된 기계어 모아놓은 것
- namespace
    - 모듈을 호출할 때 범위 정하는 방법
    - from 과 import 사용
    - (추천) alias 설정 → import tensorflow as tf
    - 모듈에서 특정 함수 또는 클래스만 호출하기 → from aa import a
    - 모듈에서 모든 함수 또는 클래스 호출 → from aa import *
- random 등 다양빌트인 모듈 존재함 → 모듈 키워드를 알고 어떻게 사용할지 생각하는 것 중요

## 패키지

- 모듈을 모아놓은 단위, 하나의 프로그램
- 모듈이 쌓이면 패키지, 패키지가 쌓이면 프로젝트
- 하나의 대형 프로젝트를 만드는 코드의 묶음, 다양한 모듈들의 합, 폴더로 연결됨
- __init__, __main__ 등 키워드 파일명이 사용됨
- 다양한 오픈 소스들이 모두 패키지로 관리됨
- 각 폴더마다 __init__ 가 있어야함
    - sound 폴더의 __init__.py

    ```python
    __all__ = ["bgm", "echo"]

    from . import bgm
    from . import echo
    ```

- 폴더 자체를 실행시키기 위해 __main__ 사용
    - Game 폴더의 __main__.py

    ```python
    from sound import echo

    if __name == "__main__":
    	print("Hello Game")
    	print(echo.echo_play())
    ```

    - 터미널에서 python Game 하면 실행됨
- 패키지 내에서 다른 폴더의 모듈을 부를 때
    - 상대 참조
        - .render import render_test
    - 절대 참조
        - game.graphic.render import render_test

## 가상환경 설정

- 어떤 프로젝트 목적에 맞춰서 필요한 패키니만 설치하는 환경
- virtualenv + pip 와 conda 가 있음
- 가상환경 실행 : conda activate <가상환경>
- 가상환경에 패키지 설치 : conda install <패키지>

<br>

<hr>

<br>

# 피어 세션

<br>

1. 오늘 강의 내용은 다 이해 못해서 다음주 월요일에 공부해서 질문하기로 하였다.
2. 저번 과제에 대한 코드 리뷰
    - import string 해서 '0123456789' 나 'abcd...xyz' 등 다양한 모듈 사용하면 편하다.
    - str 같이 예약어로 변수 이름 만들지 말자.
    - 중복 숫자 찾고 걸러내고 할 때, set 교집합 등 이용하면 좋다.

<br>

<hr>

<br>

# Today I Felt

<br>

## 모르면 알때까지

자바를 사용할 때는 자바가 OOP 임을 인지하며 프로그래밍 했다. 하지만 이제까지 파이썬을 쓰면서 OOP 이라는 특성을 살리면서 코딩하지 않았다. 대충 그런 특성이 있다~ 라고만 알고 적용하려 하지 않았다. 정말 부끄러운 것은 캡스톤 프로젝트를 하면서 클래스부터 모더레이터까지 모든게 사용되었지만 공부하려하지 않았다는 것이다.  
며칠 안됐지만 이제 모르는게 있으면 확실히 알기 위해 공부하고 블로깅하는 습관을 들이고 있다. 이런 습관에 대한 필요성은 피어 세션을 하면서 얻게 되었는데 이 부분에 대해 정말 부캠에 감사하다.

<br>

## 코드 리뷰와 성장

이번 과제는 더 파이썬스럽게, 더 효율적으로 짜기 위해 고민을 했다. 오늘 피어 세션에서 동료들과 저번 과제에 대한 코드 리뷰를 해본게 도움이 됐다.  
세션을 하며 동료들의 좋은 코드를 보며 감탄했다. 그리고 내 코드를 보여주고 설명하는게 힘들었다. 여러모로 코드 리뷰는 나를 성장시키는 것 같았다. 앞으로 리뷰어일때는 좋은 정보를 주고, 리뷰이일때는 겸손하게 피드백을 받아들이려 노력해야겠다.
