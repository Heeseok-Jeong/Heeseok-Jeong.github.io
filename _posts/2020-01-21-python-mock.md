---
layout : post
title : Python Mock for unittest
subtitle : 유닛테스트 지원사격 
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
- [사용법](#사용법)
- [결론](#결론)


<br>
<hr>
<br>

# Mock 이란?
<br>

유닛 테스트를 수행할 때 단순히 맞고 틀린지 등을 비교하는 `assert` 계열의 함수만으로 테스트가 충분하지 않을 때가 있다. 테스트를 위해 외부 API 가 필요하거나 복잡한 테스트 환경이 필요할 때, `mock` 을 import 하여 테스트를 진행하면 된다.  
<br>

`mock` 의 원래 뜻은 모조품, 모조 객체이다. 즉, `mock` 은 테스트에 필요한 객체를 직접 사용하지 않고 대신 그 객체인척 행동해주는 기특한 모듈인 것이다. 

<br>
<hr>
<br>

# 사용법
<br>

1. decorator
    `@mock` 데코레이터를 통해 객체를 mock 으로 치환시킬 수 있다. 
    ```python
        

    ```
2. context manager
3. inline