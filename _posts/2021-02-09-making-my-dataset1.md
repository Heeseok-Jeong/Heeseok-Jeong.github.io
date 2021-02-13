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

# 포스팅중...

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

## 조유리즈 (사람이 봐도 헷갈리는 데이터)

처음 선정했던 주제는 조유리즈다. 조유리즈란 

<br>
<hr>
<br>

# 데이터 수집
<br>


<br>
<hr>
<br>

# 데이터 가공
<br>


<br>
<hr>
<br>

# 데이터로 dataset 만들기 
<br>



