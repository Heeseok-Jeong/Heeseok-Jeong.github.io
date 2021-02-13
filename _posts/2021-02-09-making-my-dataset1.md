---
layout : post
title : 나만의 데이터셋으로 이미지 분류하기(1) 
subtitle : 데이터셋 만들기
tags : [Machine Learning]
author : Heeseok Jeong
comments : True
use_math: True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목록

- [개요](#개요)
- [주제 선정](#주제-선정)
- [데이터 수집](#데이터-수집)
- [데이터 가공](#데이터-가공)
- [데이터로 dataset 만들기](#데이터로-dataset-만들기)

<br>
<hr>
<br>

# 개요  
<br>

파이토치로 모델을 학습하기 위한 데이터는 `torch.utils.data.Dataset` 객체로 변환되어 `torch.utils.data.DataLoader` 에 올라가야 모델에 학습이 가능하다.  
<br>

일전에 이미지 분류 모델을 학습하기 위해 파이토치에서 제공하는 `MNIST` dataset 을 사용하였다 `torchvision.datasets.MNIST`.  
구체적으로는 `Vanilla CNN` 모델 + `CrossEntorpyLoss` 로스 함수 + `Adam` optimizae + `MNIST` 데이터를 사용하여 모델을 학습하였다.  
<br>
  
이제부터는 부스트캠프 조원들과 함께 직접 데이터 셋을 만들어서 이미지 분류 모델에 학습하고자 한다. 이번 포스트에서는 데이터셋 구축까지의 절차를 담을 것이다.   
<br> 
  
> 절차  
  1) 주제 선정  
  2) 데이터 수집  
  3) 데이터 가공 (데이터 완성)  
  4) 데이터로 dataset 만들기

<br>
<hr>
<br>

# 주제 선정
<br>

### 조유리즈 (사람이 봐도 헷갈리는 데이터)

처음 선정했던 주제는 조유리즈다. 아이즈원의 멤버 김채원, 조유리, 최예원은 팬들이 봐도 헷갈릴정도로 비슷하게 생긴 멤버들이다.

![image1]({{ site.baseurl }}/assets/img/making_my_dataset1/1.png)

(정말 닮았다)

=> 데이터가 적기 때문에 우선은 확실히 구분가는 레이블을 주제로 변경

<br>
<br>

### 마동석, 수지, 유재석 (구분하기 쉬운 데이터)

인터넷에 사진이 많고 특징이 뚜렷한 인물들로 주제를 변경하였다. 조 인원이 6명이기 때문에 2 명씩 한 인물에 대해 데이터를 모으기로 하였고 나는 수지를 담당하였다.    

<br>

추가적으로 데이터 양은 한 인물마다 학습이 잘 될 것 같은 사진 300 장을 팀별로 선별하였다.


<br>
<hr>
<br>

# 데이터 수집
<br>

## How?

이미지 데이터를 모으기 위해서는 인터넷에 존재하는 여러 플랫폼에서 이미지를 가져와야 한다. 이미지를 가져오는 방법에는 직접 하나하나 다운로드할 수도 있지만, 이를 효율적으로 하기 위해 크롤링 툴을 사용하였다.

### 구글 이미지 다운로드 - googleimagesdownload

가장 많은 정보가 존재하는 구글에서 이미지를 가져오는 프로그램은 이미 존재한다! 다음 프로그램을 이용하여 이미지를 가져오자.

<br>

-> [google_images_download](https://github.com/hardikvasa/google-images-download)

> 위 프로그램을 사용할 때 100장 이상은 크롤링이 필요하므로 `chromedriver` 를 설치하고 --chromedriver 옵션에 설치한 크롬 드라이버 위치를 명시해야한다.  

<br>

'수지', '미쓰에이 수지', '배수지', '수지비주얼' 키워드로 각 키워드 당 350 여장의 사진을 다운로드하였다.

<br>

### 인스타 이미지 다운로드 with tag

SNS 중 사진이 가장 많은 인스타그램에서 원하는 # 태그를 검색하고 첫번째로 나오는 태그에서 원하는만큼 사진을 가져오는 프로그램을 이용하였다 (구글링하여 코드 작성).

<br>

-> [instagram_image_download](https://github.com/Heeseok-Jeong/instagram_image_download)

<br>

'배수지' 키워드로 350 여장의 사진을 다운로드하였다.

<br>

### 수집 데이터

전체 1700 여장의 사진을 수집하였다.

<br>
<hr>
<br>

# 데이터 가공
<br>

인터넷에서 수집한 사진들 중 학습에 적합한 사진을 골라내는 단계이다. 매뉴얼하게 일일이 골라내는 방법을 사용하였다. 아래의 룰을 기반으로 사진을 삭제하였다.

- 연예인 수지와 관계 없는 사진
- 정면 얼굴이 뚜렷하지 않은 사진
- 누군가와 함께 있는 사진

<br>

과정을 수행하여 1700 장의 사진에서 300 장의 사진 (25%) 만 추려내었다. 


<br>
<hr>
<br>

# 데이터로 dataset 만들기 
<br>

현재 '동석', '수지', '재석' 폴더 (레이블) 에 약 300 장의 이미지 파일들이 존재한다. dataset 을 만드는 것은 전체 데이터를 일정 비율로 나눠 train 과 test 로 분리하는 것을 의미한다.  

<br>

먼저 jpg, png 이미지 파일들이 섞여있기 때문에 파일들을 읽어 모두 png 파일로 변형한다.  
`sklearn 의 train_test_split` 함수를 통해 레이블마다 train 과 test 데이터 비율을 일정하게 만들어준다. 그리고 이 파일들을 `dataset/train` 과 `dataset/test` 폴더에 넣어 dataset 을 준비한다.  

```python
from sklearn.model_selection import train_test_split

train_image, test_image, train_target, test_target = train_test_split(dataset[:,0], dataset[:,1], stratify=dataset[:,1],)
```

<br>

=> 나만의 데이터셋을 수집하고 가공하여 모델 학습에 필요한 dataset 완성
![image2]({{ site.baseurl }}/assets/img/making_my_dataset1/2.png)
![image3]({{ site.baseurl }}/assets/img/making_my_dataset1/3.png)
