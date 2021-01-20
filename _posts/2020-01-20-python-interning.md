---
layout : post
title : Python 에서 -5 부터 256 까지의 메모리 저장 방식 
subtitle : interning 방식
tags : [Python]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

- [파이썬 메모리 저장 방식](#파이썬-메모리-저장-방식)
- [결론](#결론)

<br>
<hr>
<br>

# 파이썬 메모리 저장 방식  
<br>

파이썬에서는 모든 것이 객체이다. `a = 500` 라는 변수를 선언하면 a 라는 객체가 생기고 500 이 힙 영역의 한 메모리에 할당된다. 이후 a 라는 객체는 500 을 저장하는 메모리 주소를 가리키게 된다.  
<br>
  
이제부터 신기한 파이썬만의 메모리 저장방식이 등장한다. `a = 500`, `b = 500` 는 일반적인 소프트웨어 시점에서는 같은 상수를 저장하고 있으므로 같다고 느껴진다. 파이썬에서도 값만 비교하는 연산자인 `==` 를 사용하면 True 가 리턴된다. 하지만 값과 메모리를 둘 다 체크하는 연산자인 `is` 를 사용하면 `a is b` 는 False 가 리턴된다. 500 이라는 값이 서로 다른 메모리에 할당되어 a 와 b 는 다른 메모리 주소를 가리키기 때문이다.  
<br>
  
하지만 위 과정을 -5~256 과 [a-zA-Z0-9_] 에 대해서 수행하면 `a is b` 는 True 를 리턴한다. 왜그럴까? 정답은 `interning` 이라는 방식 때문이다. 해당 숫자와 문자열은 자주 사용하기 때문에 언어를 만들 때부터 cPython 코드에서 이미 지정된 주소를 할당하여 어떤 변수가 사용하더라도 지정된 메모리 주소를 주는 방식이 `interning` 이다. 즉, a 와 b 가 5 를 저장하면, a 와 b 모두 같은 메모리를 가리키게 되는 것이다. 
<br>
<br>

![Pass Image]({{ site.baseurl }}/assets/img/python_interning.png)

<br>

위 사진을 보면 256 까지는 지정된 메모리 주소를 따라 2*16 = 32 비트 (4바이트) 씩 메모리 주소가 커지는 반면, 다른 숫자에 대해서는 같은 메모리를 할당하는 것을 볼 수 있다 (변수가 없어서 같은 메모리 할당) .

<br>
<hr>
<br>

# 결론  
<br>

> 파이썬은 -5~256 과 [a-zA-Z0-9_] 에 대해서는 어떤 변수가 사용해도 같은 메모리를 할당하는   
`interning` 방식을 사용한다.

<br>

