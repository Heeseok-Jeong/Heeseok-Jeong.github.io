---
layout : post
title : Ustage Day 5
subtitle : 파이썬 File|Exception|Log Handling|Data Handling
tags : [BoostCamp AI Tech]
author : Heeseok Jeong
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

<br>

- [File / Exception / Log Handling](#file---exception---log-handling)
- [Python Data Handling](#python-data-handling)
- [마스터 세션 - 최성철 마스터님](#마스터-세션---최성철-마스터님)
- [피어 세션](#피어-세션)
- [느낀점](#느낀점)

<br>

<hr>

<br>

# File / Exception / Log Handling

## Exception

- 예상 가능한 예외
    - 발생 여부를 사전에 인지가능
    - 사용자의 잘못된 입력, 파일 호출 시 파일 없음 등
    - 개발자가 명시적으로 처리해야 함

    → if 문으로 처리 가능

- 예상 불가능한 예외
    - 인터프리터 과정에서 발생하는 예외, 개발자 실수
    - 리스트의 범위를 넘어가는 호출, 정수 0으로 나눔
    - 수행 불가시 인터프리터가 자동 호출

    → Exception Handling 으로 처리

<br>

### 예외 처리 (Exception Handling)

- 예외에 적절한 처리를 함
- try ~except 문법

    ```python
    try:
    	예외 발생 가능 코드
    except <Exception Type>:
    	예외 발생시 대응 코드
    

    for i in range(10):
    	try:
    		print(10 / i)
    	except ZeroDivisionError: # ZeroDivisionError 는 빌트인 에러
    		print("Not divided by 0")
    ```

    - 익셉션 발생해도 프로그램 끝나지 않음, 안정적으로 프로그램 수행
    - 맨 밑에 `except Exception` 하는건 어떤 익셉션인지 알기 어렵기 때문에 좋지 않음
- finally 구문 : 예외 상관없이 항상 실행
- raise 구문 : 필요에 따라 강제로 익셉션을 발생
- assert 구문 : 특정 조건을 만족하지 않을 경우 예외 발생

<br>

## File Handling

- 텍스트 파일 (사람이 이해하는 문자열 형태, 메모장으로 볼 수 있음) 과 바이너리 파일 (컴퓨터가 이해하는 이진 코드 형태/ 엑셀 파일, 워드 파일 등등 메모장으로 못봄) 로 나뉨
    - 텍스트 파일도 실제는 바이너리 파일, 아스키 코드나 유니코드로 변환시킨 것
- File Read

    ```python
    # read() txt 파일 안에 있는 내용을 문자열로 반환
    f = open("book.txt", "r") # 상대 경로
    contents = f.read()
    print(contents)
    f.close()

    # with 구문으로 읽기
    with open("book.txt", "r") as my_file:
    	contents = my_file.read()
    	print(type(contents), contents)
    ```

    - read() : 전체를 하나의 문자열로 담음
    - readline() : 한 줄을 문자열로 담음
    - readlines() : 문자열 한 줄씩 리스트에 담아줌
- File Write

    ```python
    # mode = "w", encoding = "utf8"
    f = open("book.txt", "w", encoding="utf8) 
    for i in range(1, 11):
    	data = "%d번째 줄입니다.\n" % i
    	f.write(data)
    f.close()

    # with 구문으로 읽기, mode = "a"
    with open("book.txt", "a", encoding="utf8") as my_file:
    	for i in range(1, 11):
    	data = "%d번째 줄입니다.\n" % i
    	f.write(data)
    ```

    - 인코딩 : 컴퓨터에 문자 저장하는 표준
    - w 모드 : 해당 문서를 새로 작성
    - a 모드 : 해당 문서가 없으면 만들어서 작성하고 있으면 뒤에 연결하여 작성

<br>

### OS 모듈

- 폴더 만들기

    ```python
    import os

    # 폴더 생성
    os.mkdir("book_folder")

    # 폴더 없다면 생성
    try:
    	os.mkdir("abc")
    	except FileExistError as e:
    		print("Already created")

    ```

<br>

### shutil 모듈

- shutil.copy : 파일 복사 함수

    ```python
    import shutil

    source = "book.txt"
    dest = os.path.join("abc", "book2.txt") # 옮길 경로 지정, abc/book2.txt
    shutil.copy(source, dest) # source 의 내용을 dest 로 옮김
    ```

<br>

### pathlib 모듈

- 최근에 많이 사용, path 를 객체로 다룸

    ```python
    import pathlib

    cwd = pathlib.Path.cwd() # 현재 path 위치 알려줌
    cwd.parent # 하나 윗경로 알려줌
    ```

- 등등 기타 모듈 필요할 때 찾아서 쓰면 됨

<br>

## Pickle

- 파이썬의 객체를 영속화(persistence) 하는 빌트인 객체
- 데이터, object 등 실행중 정보를 저장 → 불러와서 사용
- 저장해야하는 정보, 계산 결과(모델) 등 활용이 많음
- 파이썬에 특화된 바이너리 파일!

```python
import pickle

# 데이터를 피클로 저장해둠
f = open("list.pickle", "wb")
test = [1, 2, 3, 4]
pickle.dump(test, f)
f.close()

# test 삭제
del test

# 이제 test 데이터가 없지만, 저장해둔 피클을 불러와서 사용할 수 있음
f = open("list.pickle", "rb")
test_pickle = pickle.load(f)
print(test_pickle)
f.close()
```

- 클래스의 인스턴스도 저장 가능

<br>

## Logging Handling

- 게임에서 핵쓰는 유저 어떻게 잡을까?
    - 기록을 남겨야 잡을 수 있음! → 로그
- 로그 남기기 - Logging
    - 프로그램이 실행되는 동안 일어나는 정보를 기록으로 남기기
    - 유저의 접근, 프로그램의 익셉션, 특정 함수의 사용
    - 콘솔에 출력하거나, 파일에 남기거나, db 등에 남길 수도 있음
    - 기록된 로그를 분석하여 의미있는 결과를 도출할 수 있음
    - 실행시점에 남겨야하는 기록과 개발시점에 남겨야하는 기록으로 나뉨
- print v logging
    - 둘다 기록 남기는건 가능하지만, print 는 콘솔 창에만 남으므로 분석시 사용 불가
    - 로깅은 레벨별(개발, 운영), 모듈별 등 체계적으로 기록을 남길 수 있음
- logging level
    - 프로그램 진행 상황에 따라 다른 레벨의 로그를 출력
    - DEBUG > INFO > WARNING > ERROR > CRITICAL(제일 심각) 단계가 있음
    - 기본 로깅 레벨은 워닝 ~ 크리티컬, 이 레벨에 있는 로깅만 사용자에게 나옴 → 로깅 레벨 변경 가능
- 파일에 기록

    ```python
    import logging

    if __name__ = "__main__":
    	logger = logging.getLogger("main")
    	logging.basicConfig(level=logging.DEBUG)
    	logger.setLevel(logging.INFO)

    	steam_handler = logging.FileHandler(
    		"my.log", mode="w", encodin="utf8")
    	logger.addHandler(steam_handler)

    	logger.debug("틀림")
    	logger.error("에러")
    ```

- 로깅 전에 설정해야 할게 많음 → 툴 사용
    1. configparser
        - 프로그램의 실행 설정을 파일에 저장함
        - 섹션, 키, 밸류 값의 형태로 설정된 설정 파일 사용
        - 설정파일을 딕트 타입으로 호출 후 사용
        - example.cfg 라는 컨피그 파일이 있으면 이것을 불러와서 설정
    2. argparser (= Command-Line Option)
        - 콘솔창에서 프로그램 실행시 세팅 정보를 저장
        - ex) ls —help 에서 — 붙여서 사용
        - add_argument 로 추가 가능

        ```python
        import argparse

        parser = argparse.ArgumentParser(
        	description="Sum two integers.")

        parser.add_argument(
        	'-a', "--a_value",
        	dest="a", help="A integers", type=int,
        	required=True)
        ```

    <br>

    <hr>

    <br>

# Python Data Handling

<br>

## CSV (Comma Separate Value)

- 필드를 쉼표로 구분한 텍스트 파일
- 엑셀 양식의 데이터를 프로그램에 상관없이 쓰기 위한 데이터 형식
- 콤마 말고 탭(TSV), 빈칸(SSV) 등으로 구분하기도 함 → 모두 CSV 라고 부름
- 파일 읽고 콤마로 split 하여 사용 → 더 간단히 csv 파일 처리하기 위해 csv 객체 사용하자
    - delimeter : 글자를 나누는 기준
    - lineterminator : 줄 바꿈 기준
    - quotechar : 문자열을 둘러싸는 신호 문자
    - quoting : 데이터 나누는 기준이 quotechar 에 의해 둘러싸인 레벨

    ```python
    import csv

    reader = csv.reader(f, delimeter=', ',
        quotechar='"', quoting=csv.QUOTE_ALL)

    ```

- 첫 줄은 데이터 헤더, 나머지는 일반 데이터
- 한글 파일 읽을 때 encoding='cp949'

<br>

## 인터넷 (web)

- WWW (World Wide Web), 줄여서 웹
- 우리가 늘 쓰는 인터넷 공간의 정식 명칭
- 데이터 송수신을 위한 HTTP 프로토콜 사용
- 데이터 표시를 위해 HTML 형식 사용
- 웹은 어떻게 동작하는가?
    1. 요청 : 웹주소, Form, 헤더 등 
    2. 처리 : 데이터베이스 처리 등 요청 대응
    3. 응답 : HTML, XML 등으로 결과 반환
    4. 렌더링 : HTML, XML 표시
- 왜 웹을 알아야 하는가?
    - 많은 데이터들이 웹을 통해 공유됨
    - HTML 도 일종의 프로그램, 페이지 생성 규칙이 있음 → 규칙을 분석하여 데이터 추출 가능
    - 추출된 데이터로 다양한 분석 가능

<br>

### HTML (Hyper Text Markup Language)

- 웹 상의 정보를 구조적으로 표현하기 위한 언어
- 제목, 단락, 링크 등 요소 표시를 위해 Tag 사용

<br>

### 정규식 (regular expression)

- 정규 표현식, regexp 또는 regex 등으로 불림
- 복잡한 문자열 패턴을 정의하는 문자 표현 공식
- 특정한 규칙을 가진 문자열의 집합을 추출
- 주민등록 번호, 전화번호, 도서 ISBN 등 형식이 있는 문자열 파싱
- 기본적인거만 공부하고 필요한거 찾아서 사용
    - 정규식 연습장 ([https://regexr.com/](https://regexr.com/)) 에서 연습 가능
- 기본 문법 #1
    - 문자 클래스 [ ] : [ 와 ] 사이의 문자들과 매치라는 의미
        - [abc] : 해당 글자가 a, b, c 중 하나가 있다.
    - '-' 를 사용하여 범위를 지정할 수 있음
        - [a-zA-Z] : 알파벳 전체
    - 원래 의미 말고 정규식용 의미 지닌 문자들
        - . ^ $ * + ? { } [ ] \ | ( )
    - . : 줄바꿈 문자 \n 제외한 모든 문자와 매치
    - * : 앞에 있는 글자를 반복해서 나올 수 있음
    - + : 앞에 있는 글자를 최소 1회 이상 반복
- (http)(.+)(zip) : 특정 페이지에서 http~.zip 추출하기

<br>

### 정규식 in Python

- re 모듈을 import 하여 사용 `import re`
- 함수 : search - 한 개만 찾기, findall - 전체 찾기
- 추출된 패턴은 tuple 로 반환
- ([A-Za-z0-9]+\*\*\*) : 특정 페이지에서 ID만 추출하기 (dldydld***)
- 파이썬 코드

```python
import re
import urllib.request

url = "https://bit.ly/3rxQFS4"
html = urllib.request.urlopen(url)
html_contents = str(html.read())
id_results = re.findall(r"([A-Za-z0-9]+\*\*\*)", html_contents)

for result in id_results:
	print(result)
```

<br>

## XML

- 데이터의 구조와 의미를 설명하는 TAG(MarkUp) 을 사용하여 표시하는 언어
- TAG 와 TAG 사이에 값이 표시되고, 구조적인 정보를 표현할 수 있음
- HTML 과 문법이 비슷, 대표적인 데이터 저장 방식
- XML 은 컴퓨터 간에 정보를 주고받기 매우 유용한 저장방식으로 쓰이고 있음
- 정보의 구조에 대하 정보인 스키마와 DTD 등으로 정보에 대한 정보(메타정보)가 표현되며, 용도에 따라 다양한 형태로 변경가능

<br>

### XML Parsing in Python

- XML 도 HTML 과 같이 구조적 markup 언어
- 정규표현식으로 Parsing 이 가능하지만, 더 쉬운 `beautifulsoup` 으로 파싱
- lxml 파서도 사용

<br>

## JSON (= Javascript Object Notaion)

- 원래 웹 언어인 js 의 데이터 객체 표현 방식
- 간결성으로 기계/인간이 모두 이해하기 편함
- 데이터 용량이 적고, 코드로 전환이 쉬움
- xml 대체로 많이 사용 → xml 보다 간결하고 용량도 적음
- dict 처럼 키:밸류 쌍으로 표시
- import json 해서 사용 → load 하면 자동으로 dict 로 받아들임
- read 는 load, write 는 dump

<br>

<hr>

<br>

# 마스터 세션 - 최성철 마스터님

<br>

## QnA

### 1. AI 로 할 수 있는 일이 어떤게 있을까요?

- NLP
- 데이터마이닝
- 데이터 클러스터링 (비지도학습)
- 영상처리 등

### 2. 만약 AI 를 더 공부하고 싶어서 대학원을 간다면 어떤 것을 더 준비해야 할까요?

- 수학 : 선형대수, 미적분, 통계학, 수리통계, 해석학
- 프로그래밍 능력 : 문제해결 능력, db, 엑셀 등
- 석사 → 논문을 읽고 구현하는 능력이 있어야 함

### 3. 수학을 배운지 오래된 학생인데 AI 를 제대로 배우기 위해서는 선형대수, 미적분, 기초통계학에 대한 지식이 있어야 된다고 들었습니다. 지금부터 공부하려고 하는데 세 과목을 어느 수준으로 공부해야 앞으로 과정을 따라가기가 수월할까요?

- 논문 이해하는데 문제 없을 정도
- 책에 문제가 있다면 반은 풀 정도

### 4. 강화학습을 활용한 자율주행이나 게임인공지능 분야에 관심이 많습니다.

1) 강화학습 (RL) 이 쓰이는 분야는 어디가 있나요?

- 게임, 싸펑은 RL 로 학습해서 성능이 떨어지는게 아닌가
- offline RL

2) AI 분야의 데이터 엔지니어를 희망하면 수학은 어느정도로 해야 할까요?

- 3 에서 언급

3) 강화학습과 딥러닝 비교시 산업별 장단점은 무엇인가요?

- 뒤에서 언급할 예정

4) 강화학습 프로젝트 경험이 있으신가요?

### 5. AI 에서 특별히 많이 쓰이는 pythonic 코드가 있을까요? Str 함수 외에도 궁금합니다.

- 대박 같은 slang 처럼 일상적으로 쓰임
- list comprehension, zip, enumerate

### 6. 데이터 사이언티스트와 머신러닝 엔지니어는 어떻게 다른지 알고 싶습니다.

- 머신러닝 엔지니어
    - 머신러닝, 딥러닝에 대해 잘 이해하고 논문을 읽고 기존 모델을 전체 시스템과 잘 연동한 파이프라인을 만드는가
    - 새로운 모델을 만드는 일 x, 활용하고 서비스 하는 일
    - 서비스까지에 있어 데이터 전처리 하고 리소스를 위해 경량화하는 등
- 데이터 사이언티스트
    - 거의 모든 분야 다루는데 좁게 말하면 데이터를 분석해서 의사결정에 도움을 준다거나 하는 일 (스타트업은 잘 안함)
    - 인터넷에서 Full-stack DL 이라는 강의 듣기 권장

### 7. 수강 관련 질문 - 뒤에서 말할 것

### 8. 교수님이 현재 부스트캠퍼라면 캠프 과정을 따라가는 것 이외에 무엇을 하실지 궁금합니다.

- 주식 → 세상에 관심을 갖기 위해

### 9. 읽기 좋은 코드는 꼭 간결해야 하나요?

- 간결하다고 꼭 좋은 코드는 아님

### 10. AI 서비스를 만들기 위해 필요한 엔지니어링 기술을 배우고 싶습니다. 빅데이터 처리에 스파크를 쓰는 것처럼, 배우면 좋을 기술 스택이 있을까요?

- 스파크도 쓸 줄 알아야 하고, 하둡, 에어 플로우 등 다 관심깊게 보는게 좋다.

### 12. 미래에 AutoML 이 발전한다면 모델링도 AutoML 이 해결해줄거라 예상하나요?

- 딱히 그렇게 안 봄, 전체 파이프라인을 연결해주는 능력 + 서비스 만들어내는 능력은 사람의 중요한 역량
- 엔지니어라면 파이프라인쪽(앞단) 역량 중요

### 13. AI를 공부하고 관련 직종에 취직하기 위해서는 아무래도 대학원 석사과정을 밟는 것이 거의 필수적인가요??

- 필수는 아닌데 확률을 매우 높이기 때문에 추천함
- 박사는 별로 권장안함, 석사는 괜찮다~
- AI 엔지니어 할거면 석사 정도는 돼야한다 생각
- 백엔드(웹개발) 역량은 교양처럼 쓰인다~
- 모델을 개발할 때는 파이썬 쓰지만, 실제 서비스 할 때는 C 나 자바 계열 많이 씀

### 14. 파이프라인 짜는 능력은 어떻게 키울 수 있나요?

- 프로젝트를 통해서만 키울 수 있다.

<br>

<hr>

<br>

# 피어 세션

<br>

## 일주일 회고

- 후미 : 생각보다 1주차가 널널하지 않고 빡셌다.
- 펭귄 : 팀프로젝트에 대한 트라우마가 있어서 긴장했는데 좋은 조원들 만나서 좋다. 자바나 씨쁠쁠보다 파이썬 되게 좋아하는데 이해 못하고 잘 몰랐던 부분 조금 더 깊이 공부할 수 있어서 좋았다.
- 원딜 : 예전부터 ML 하면서 파이썬 써왔는데 모르는 부분 얻을 수 있어서 좋았다. 전에 강의들을때보다 더 자세하게 공부할 수 있어서 좋았다.
- 서폿 : 팀원에 대한 걱정이 많았는데 좋아서 다행이고 감사하다. 코드스쿼드랑 부캠 동시에 합격해서 선택해야 했는데 여러가지 + 조원들 보고 부캠 선택했다. 강의적으로도 파이썬에 많은 도움이 되고 있다. 파이썬 코딩 도장 책 추천한다.
- MJ : 조원들 때문에 피어세션이 기다려지고 힘이 난다. 펭귄님 리딩 덕에 코드 리뷰 했던 것이 많은 도움이 됐다.
- 샐리 : 펭귄님이 첫 틀을 잘 잡아줘서 감사하고 피어세션 이 수업보다 재밌다. 파이썬 안다고 생각했는데 심화된 내용 들으면서 정리할 수 있어서 좋았다. 앞으로가 기대된다.

<br>

## 피어 세션 피드백

- 히스 : 오늘 수업 들은거 모르는거나 질문 그날 피어세션에 질문하고 답 못찾으면 그날 자습하면서 답 달고 다음날 체크하면 좋겠다.
→ 다음주부터 적용
- 서폿 : 알고리즘 문제 하루에 1 개씩 풀면 좋겠다.
- 일정이 조금 널널하다면, 주제를 정해서 같이 공부해도 좋을 것 같다.
- 자기가 아는 내용에 관해 5분 정도 짧게 세미나 하면 좋을거 같다. (일주일 한 번 정도)
    - 서폿 : 세미나를 더 키우자 (슬랙에 올려서 일주일 15분 3명 정도로, 비어세션)
    - 후미 : 판이 커지면 어수선해지고 그럴거 같다.

    세부 사항

    - 1 인 최대 10분 발표 + QnA 시간 최대 10분
    - 형식 : 자유롭게
    - 주제 : 자유롭게
    - 시간 : 수요일 피어세션 시작부터

## 코드 리뷰 - 과제 2

- 원딜님 코드 : any 를 써서 원소 걸러내는 방법 익히는 방법 인상적
- 서폿님 코드 : 딕트 밸류 반대로 만드는법

    ```python
    r_map = dict(zip(o_dict.values(), o_dict.keys()))
    ```

<br>

<hr>

<br>

# 느낀점

<br>

 

## 환기

오늘 피어 세션에서 펭귄님이 각자 자유롭게 세미나 하는 시간을 가져보자는 의견을 내주셨다. `우아한 테크 톡` 을 보면서 저런거 해보면 재밌겠다 싶었기에 펭귄님의 제안이 신선했다. 멋지다 생각만 하고 실천까지 생각하지 못하던 것들을 조원들을 통해 할 수 있다니..  

혼자 였으면 고일 뻔한 물이 다른 사람들로 인해 환기되고 있다. 부캠이 끝나면 나는 어떻게 성장했을지 설렌다.  
<br>

## 완전한 이해는 어려워

피어 세션을 하면서 MJ님이 리스트가 아이터레이터인데 왜 또 iter 라는 내장함수가 있냐는 질문을 했다. 다행히 내가 공부했던 부분이기에 (원래였다면 안했겠지? 대견스) 최대한 아는 선에서 바른 답을 알려드리려 노력했다. 이렇듯 모르는 부분에 대해 끝까지 공부하는 습관이 도움이 되고 있다. 하지만 `mock` 에 대한 개념이 어려워서 며칠째 어떻게 정리해서 블로깅할지 머리가 아프다. 솔직히 안올리고 싶은 마음이 굴뚝스,, 그치만 확실한 이해가 도움이 됐음을 상기하며 주말 내에 꼭 이해해서 포기하지 않고 글을 올려야겠다.
