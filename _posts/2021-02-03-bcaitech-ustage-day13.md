---
layout : post
title : Ustage Day 13
subtitle : CNN|1x1 convolution
tags : [BoostCamp AI Tech]
author : Heeseok Jeong
comments : True
use_math: True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차

<br>

- [CNN - Convolution은 무엇인가?](#cnn---convolution은-무엇인가-)
- [Modern CNN - 1x1 convolution의 중요성](#modern-cnn---1x1-convolution의-중요성)
- [Computer Vision Applications](#computer-vision-applications)
- [CNN 실습](#cnn-실습)
- [피어 세션](#피어-세션)
- [Today I Felt](#today-i-felt)

<br>

<hr>

<br>

# CNN - Convolution은 무엇인가?

<br>

## Convolution

- 수식
    - 연속형

        $(f*g)(t) = \int f(\tau)g(t-\tau)d\tau = \int f(t-\tau)g(t)d\tau$

    - 이산형

        $(f*g)(t)=\sum_{i=-\infty}^{\infty}f(i)g(t-i)=\sum_{i=-\infty}^{\infty}f(t-i)g(i$)

- convolution 은 f 와 g 함수를 잘 섞어주는 방법
- 필터로 입력값을 도장찍듯 아웃풋을 만들며 이동함
- 2D convolution 의 의미 : 2D 이미지를 여러 필터들이 찍어내면서 필터들은 초기 웨이트에 의해 특정 기능을 하는 필터가 됨
- 32x32x3 이미지에 5x5x3 필터 4개를 찍어내면 28x28x4 아웃풋이 나옴
- convolution 과정을 여러번 수행
- CNN 은 convolution layer, pooling layer, and fully connected layer 로 구성됨
    - Convolution and pooling layers : feature extraction
    - Fully connected layer : decision making (e.g., classification)
    - 최근에는 fully connected layer 를 없애는 추세, 학습시키고자 하는 모델의 파라미터 숫자가 늘어날수록 학습이 어렵고 generalization 성능이 떨어지기 때문
    - convolution 은 딥하게, 파라미터는 줄이도록 진행함

*유명한 모델들의 파라미터 표를 보고 코드를 보면서 파라미터 수가 일치하는지 확인하면서 공부할 것

### Stride

- 커널로 입력값을 옮기는 단위

### Padding

- 커널로 입력값을 찍어내면 원래 입력값의 사이즈보다 작게 나옴
- 제로 패딩, 입력 가장자리에 0 을 채워넣어 원래 사이즈대로 나올 수 있게 해줌
- 원래 사이즈대로 나오면 입력값의 가장자리를 관찰하기 좋음

### 출력값 크기

- W, H (가로, 세로) 에 똑같이 적용
- n : 입력값의 가로 혹은 세로 길이, f : 필터의 가로 혹은 세로 길이
- (n - f + 2*padding)/stride + 1

### 파라미터 개수

![image1]({{ site.baseurl }}/assets/img/ustage_day13/1.png)

- 커널의 크기 : 3x3x128
- 이 커널이 64개 존재 ⇒ 3x3x128x64 = 73,728

### 알렉스넷 파라미터 개수

- convolution 과정에서는 필터크기 * 아웃풋 뎁스 * 개수
- 마지막 dense layer (fully connected) 에서는 입력값 전체 크기 * 전체 아웃풋크기
→ 13 * 13 * 128 * 2 x 2048 * 2 ~ 177M, 컨볼루션에 사용되는 파라미터보다 훨씬 많은 파라미터 사용됨

    ![image2]({{ site.baseurl }}/assets/img/ustage_day13/2.png)

- 컨볼루션은 입력값 모든 부분에 동일한 크기로 적용하기 때문에 파라미터 개수가 적음 → 파라미터가 줄었으니 성능이 좋음

## 1x1 Convolution 중요

- 영역을 보지 않고 이미지에서 한 픽셀만 봄 + 채널 개수 줄임
- 왜할까?
    - Dimension reduction (채널을 줄임)
    - 컨볼루션 레이어를 더 깊게 쌓으면서 파라미터를 줄일 수 있음
    - e.g., bottleneck architecture

    ![image3]({{ site.baseurl }}/assets/img/ustage_day13/3.png)

### 실습

- 전체적으로 MLP 와 동일하지만 모델 정의 부분이 다름
- 과정
    - convolution 레이어 세팅, cdims 개수만큼 생성
        - self.layers = [] # 빈 레이어 생성
        - 입력차원 크기 설정, in_channels = prev_cdim # prev_cdim = self.xdim[0], xdim=[1,28,28]
        - 출력차원 크기 설정, out_channels = cdim
        - kernel_size 설정, kernel_size=self.ksize
        - 스트라이드 설정, stride=(1,1)
        - 패딩 설정, padding=self.ksize//2
        - batchnorm, activation(ReLU), max-pooling, dropout 을 layer 에 추가
    - flatten 하여 prev_hdim 에 저장
    - hdims 개수만큼 fully connected 수행
    - 모든 레이어를 연결, [self.net](http://self.net) = nn.Sequential() # 이제 포워드 한번만 하면 알아서 끝까지 감
        - net 에 아까 만들어둔 레이어 직접 넣음, self.net.add_module(layer_name, layer)
    - 파라미터 초기화
        - conv 레이어의 웨이트는 kaiming_normal_ 로 설정, bias 는 zeros_ 로 설정
        - BN 의 웨이트는 1, bias 는 0 설정
        - dense (fully connected 부분) 레이어도 웨이트는 kaiming_normal_ 로 설정, bias 는 zeros_ 로 설정
    - 모델 생성, xdim, ksize, cdims, hdims, ydim 설정
    - loss 는 크로스엔트로피, 옵티마이저는 Adam 설정
    - 만든 파라미터 체크
        - 먼저 conv 레이어의 파라미터가 나옴 00, 04 ... (중간에 빈 값들은 배치놈, 액티베이션, 옵티마이저로 사용됨)
        - dense 레이어 파라미터 체크
    - 간단한 forward 체크
    - eval 정의 및 간단한 체크
    - 학습
    - 테스트
- 기억할것
    - cdims 과 hdims 를 세팅할 수 있게 만들면 편함
    - 레이어의 이름을 세팅하면 디버깅할 때 편함

<br>

<hr>

<br>

# Modern CNN - 1x1 convolution의 중요성

<br>

## 개요

- ILSVRC, ImageNet Large-Scale Visual Rcognition Challenge, 이미지넷 대회 1등 모델들 볼 것
    - 분류, 감지, 지역화 등 다양한 문제 있음
    - 많은 데이터 있음
    - 사람의 오차율은 5.1% 인데 2015 년 오차율은 3.5%, 딥러닝이 사람보다 좋아짐
- 해봐야 몇 년 안됐지만 모던이라 하자
- 발전할수록 네트워크는 더 깊어지고 파라미터 개수는 줄어듦
- 파라미터 개수와 네트워크 뎁스를 잘 보자

## AlexNet

![image4]({{ site.baseurl }}/assets/img/ustage_day13/4.png)

- 처음에 두개의 아웃풋이 나오는 이유 : 당시 기계 성능 문제로 나눠서 진행한 것
- 처음 11x11 이 사람입장에서 좋은 숫자는 아님, 하나의 컨볼루션 커널이 볼 수 있는 영역은 커지지만 더 많은 파라미터 필요
- 5개의 컨볼루션 레이어와 3개의 덴스레이어 = 8단 레이어
- 키 아이디어
    - Rectified Linear Unit (ReLU) activation
    - 2 개의 GPU 사용
    - Local reponse normalization , Overlapping pooling
    - Data augmentation
    - Dropout

    ⇒ 요즘 흔히 사용하는 방법들을 제시한 것

- ReLU Activation

    ![image5]({{ site.baseurl }}/assets/img/ustage_day13/5.png)

    - 선형 모델의 특성 유지
    - 경사하강법으로 최적화하기 용이
    - generalization 에 좋음
    - vanishing gradient problem 해결
        - sigmoid 나 tanh 를 미분하면 범위가 [0, 1/4], 더 미분하면 계속 0 에 가까워지는 문제

## VGGNet

- 특징
    - 3x3 컨벌루션 필터 (stride = 1) 만 사용 (중요)
    - 풀리 커넥티드 레이어를 위해 1x1 컨벌루션 사용 (안중요)
    - dropout(p=0.5)
    - VGG16, VGG19 사용
- 왜 3x3 컨벌루션?
    - 컨벌루션이 사이즈가 크면 한 번에 많이 볼 수 있지만, 그것보다 사이즈를 작게하여 여러 뎁스로 더 보는게 나음 
      - Receptive field, 하나의 출력에 관여하는 입력 픽셀의 개수

    ![image6]({{ site.baseurl }}/assets/img/ustage_day13/6.png)

    - 3x3 레이어를 두 개 쌓으면 9 * 2 = 18, 5x5 레이어 한 개 쓰면 25 이므로 파라미터 개수가 더 적음

## GoogLeNet

![image7]({{ site.baseurl }}/assets/img/ustage_day13/7.png)

- 특징
    - NiN, Network in Network 구조
    - Inception blocks 활용

        ![image8]({{ site.baseurl }}/assets/img/ustage_day13/8.png)

        - 하나의 입력이 들어왔을 때 여러 개로 퍼졌다가 합쳐짐
        - 각각의 패스를 보면 3x3 Conv 하기 전에 1x1 Conv 를 사용해줌 (중요)
            - 파라미터 개수 줄여줌
            - 어떻게?
                - 1x1 convolution 은 채널방향으로 차원을 줄이는 효과가 있음

            ![image9]({{ site.baseurl }}/assets/img/ustage_day13/9.png)

            - 입력과 출력은 같지만 파라미터 개수 엄청 줄어듦

## ResNet

- 오버피팅은 주로 파라미터 개수 때문에 야기됨, 근데 레이어가 깊은 모델은 오버피팅 때문이 아니라 그냥 레이어 적절한거보다 성능이 안나옴 (학습이 잘 안됨)
- Residual connection (Identity map)

    ![image10]({{ site.baseurl }}/assets/img/ustage_day13/10.png)

    - 예측값에 x 를 더해줌
    - 차이를 학습하기를 원함
    - 덕분에 레이어 깊어져도 얕은 구조보다 학습 잘 됨
    - projected shortcut 은 차원을 맞춰주기 위해 1x1 conv 사용하는 것, 근데 simple shortcut 더 사용함
    - conv → batch norm → 활성화 구조인데 활성화 → BN 하는게 더 잘된다는 말도 있음
- Bottleneck architecture
    - 구글의 NiN 과 비슷
    - 3x3 하기 전에 1x1 로 줄이고, 3x3 하고 나서 1x1 로 늘려서 차원 낮춰줌

## DenseNet

- ResNet 은 두 값을 더해주는 구조, DenseNet 은 더하지 않고 concatenate 해줌
- 문제는 채널이 커짐 → 파라미터 숫자가 커짐 → 중간에 한 번씩 채널을 줄여줌, 1x1 conv 사용
- Dense Block : 합치기
- Transition Block : 채널 줄이기
- 간단한 분류할 때 ResNet 이나 DenseNet 쓰면 성능 잘나옴

## Summary

- VGG : repeated 3x3 blocks
- GoogLeNet : 1x1 convolution
- ResNet : skip-connetcion (원래값 더하기, 차이 학습)
- DenseNet : concatenation

**Further Question**

- 수업에서 다룬 modern CNN network의 일부는, Pytorch 라이브러리 내에서 pre-trained 모델로 지원합니다. pytorch를 통해 어떻게 불러올 수 있을까요?
    - 참고: [pytorch official docs](https://pytorch.org/docs/stable/torchvision/models.html)

    ```python
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    ```

<br>

<hr>

<br>

# Computer Vision Applications

<br>

## Semantic Segmentation

- 이미지를 픽셀마다 분류하는 문제
- 일반적인 문제는 전체를 하나로 분류했다면, 세그멘테이션은 이미지 안에 다양한 픽셀들을 각자 레이블로 분류함
- 자율주행 등에 활용, 자동차, 사람, 인도, 차도 등 구분
- 지금까지는 Fully Convolutional Network 구조 배웠음, 컨벌루션 진행하다가 flat 하고 dense 진행 후 라벨 확인
- 이제는 dense 부분을 없애고 싶음, conv로 바꾸자! Convolutionalization

    ![image11]({{ site.baseurl }}/assets/img/ustage_day13/11.png)

    - flat 하고 dense 하는거나 conv 로 바꾸는거나 파라미터 개수 같음
    - 사실은 같은거, 아무것도 달라진게 없음
    - 왜할까?
        - 원래는 결과가 확률분포 식으로 단순 분류할 수만 있었다면, 이제는 히트맵에 분포가 나오게 됨 → 세그멘테이션 가능
        - 해당 이미지에 고양이가 어디 있는지 나오게 됨

        ![image12]({{ site.baseurl }}/assets/img/ustage_day13/12.png)

### Deconvolution (conv transpose)

- convolution 의 역연산
    - 컨벌루션하면 디멘션이 줄어듦 5x5 → 2x2, 디컨벌루션하면 디멘션이 늘어남 2x2 → 5x5
    - 엄밀하게는 컨벌루션의 역연산이란 건 없음 (10 이라는 숫자가 원래 뭐로 만들었는지 알 수 없기 때문)
    - 나머지 부분을 다 패딩으로 만들어줌

    ![image13]({{ site.baseurl }}/assets/img/ustage_day13/13.png)

- 결과

    ![image14]({{ site.baseurl }}/assets/img/ustage_day13/14.png)

## Detection

- 이미지 안에서 어느 물체가 어디있는가? per-pixel 말고 바운딩 박스로 찾기
- 이미지 안에서 패치 (지역) 를 엄청 뽑음 (selective search)
- RCNN
    - 이미지 가져옴
    - 이미지 안에서 사물인거같은 지역 2,000 개 뽑음
    - 알렉스넷으로 각 지역에 대한 피쳐 계산
    - linear SVMs 로 분류

    → bruteforce 같음

    - 문제점 : 지역에 대해 모두 CNN 돌리는 것
- SPPNet
    - 이미지 안에서 CNN 한 번만 돌리자
    - 이미지 안에서 바운딩 박스 뽑고, 이미지 전체에 대해서 convolution 피쳐 맵을 만들어 뽑힌 위치에 해당하는 텐서만 긁어오기
- Fast RCNN

    1) 입력 이미지로 바운딩 박스 뽑음  
    2) CNN 한 번 돌려서 convolutional feature map 생성   (SPPNet 과 동일)  
    3) 각 지역에 대해 고정 길이 피쳐를 ROI pooling 으로 부터 얻음  
    4) 클래스와 바운딩 박스 regressor 두 결과를 얻어냄  

- Faster RCNN
    - 이미지를 통해 바운딩 박스를 뽑아내는 Region Proposal 도 학습을 시켜버리자  
    → Region Proposal Network (RPN) + Fast R-CNN
    - RPM
        - 어떤 이미지를 주면 어떤 지역 안에 물체가 있을지 판단
        - anchor boxes : 어떤 크기들의 물체들이 있을지 미리 아는것 (템플릿)

    ![image15]({{ site.baseurl }}/assets/img/ustage_day13/15.png)

- YOLO(v1), You Only Look Once
    - 아주 빠른 object detection algorithm
        - baseline : 45fps / smaller version : 155fps
    - RPN 을 사용하는게 아니라 이미지를 한 번에 바로 체크
        - 여러 바운딩 박스와 분류 확률 계산을 동시에 진행하여 예측함
    - 과정
        - 이미지가 들어오면 SxS 그리드로 나눔
        - 물체를 바운딩 박스로 감지 + 분류 확률에 비춰봄 → 바로 결과 나옴
            - 각 셀은 B=5 (논문기준) 개의 바운딩 박스를 예측함
                - 그 바운딩 박스들이 box probability 를 통해 쓸모 있는지 없는지 확인
                - box refinement 와 confidence 체크를 이용
            - 각 셀은 C class probabilities 를 예측함

            ⇒ 두 정보를 취합하여 박스 + 클래스를 알게됨

        ![image16]({{ site.baseurl }}/assets/img/ustage_day13/16.png)

<br>

<hr>

<br>

# CNN 실습

<br>

## 학습 목표

- RGB 이미지 데이터 다루는 방법 학습
- data augmentation 개념 학습
- 데이터 분류 모델 학습

### Self-study guide

- Dog breed 데이터셋의 다운로드부터 Dataloader 생성까지의 전 과정을 기존의 CNN 모델 파일(py) 파일 수정하여 작성해 볼 것

### 실습 내용

- 강아지 품종 사진들을 가져와서 CNN 모델에 학습

### Tips

- 실제 모델을 서비스한다치면 강아지와 사람이 함께 있을 확률이 높음. 따라서 Robust (강건한) 모델을 만들기 위해서는 데이터 자체도 사람과 함께 있는게 좋음
- 처음 사진가져오면 '이상한 문자_품종' 형식으로 파일들이 있음 → '품종' 으로 이름 변경시킴
- 그리고 품종 안 파일들도 이름을 바꿔줌
- 이번 과정은 기존과 다르게 Data Augmentation (데이터 증강) 도 수행
    - 하나의 사진을 다양하게 편집하여 보여줌 (흐리게, 잘라서 등등)
    - 강건한 모델을 만들기 위해 수행
    - 학습에서만 수행
- datasets.ImageFolder 를 통해 자동으로 사진과 레이블 매칭 가능

### 데이터 로드

- 데이터 설치 tar 형식
- 압축 해제
- 폴더 이름 변경
- 폴더 안의 파일들을 dataset 변수에 어레이로 저장시킴
- ftrain_test_split 을 통해 트레인, 테스트 이미지와 정답으로 분류함, stratify 는 트레인과 테스트의 Y 데이터 나누는 비율 똑같이 해줌

    ```python
    from sklearn.model_selection import train_test_split

    train_image, test_image, train_target, test_target = train_test_split(dataset[:,0], dataset[:,1], stratify=dataset[:,1])
    ```

- 만든 dataset 들을 로컬에 저장시킴

### CNN 학습

- ipynb 말고 py 파일로 나눠서 python3 [main.py](http://main.py) 로 실행하는거 추천
- 모델 정의
- 모델, 옵티마이저, 로스 생성
- 학습하면서 성능 좋은 모델 따로 저장

⇒ 현재 dog_breed_dataset.ipynb 와 03_CNN.ipynb 를 합치고 세 가지 py 로 나눠서 돌려볼 것

<br>

<br>

## CNN - 나만의 데이터셋 만들기

### Self-study guide

- 팀별로 수집할 데이터 주제를 선정한다.
- 구글을 통해 관련된 데이터를 다운로드 받는다.
- 같은 class의 데이터를 폴더별로 모은다.
- 해당 데이터중 관련이 없는 데이터를 삭제하거나 새로운 분류를 만들어 따로 모은다.
- CNN 모델을 만들어 학습한다.


<br>

<hr>

<br>

# 피어 세션

<br>

## TED 세미나

- 샐리님의 음성인식 요약 어플리케이션을 만든 경험
    - 전처리 (노이즈 제거 & voice  enhancement) -> stt -> 요약
    - 코로나 때문에 현강이 별로 없어 데이터를 많이 못모음

## 수업 질문

1. [[히스] Gradient Vanishing 해결을 위한 ReLU 와 ResNet](#https://github.com/boostcamp-ai-tech-4/peer-session/issues/52)
2. [[히스] CNN 레이어가 깊을 때 각 커널의 역할](#https://github.com/boostcamp-ai-tech-4/peer-session/issues/53)

## TMI 세션

- 나의 TMI

<br>

<hr>

<br>

# Today I Felt

<br>

## 우선순위 판단

수업 정리, 실습, 추가적으로 공부할 것들, 읽어야 할 것들이 쏟아지고 있다.  

<br>

나만의 데이터셋 만들기 과정 중 구글 이미지 다운로드 프로그램을 ssh 로 코랩에 접속하여 사용하니 파이썬 버전 때문인지 잘 되지 않아서 내 로컬 컴퓨터로 다운받고 드라이브에 올리기로 노선을 바꿨다. 할 일이 이 것 뿐이었으면 괜찮았을텐데 남은 일들이 많아 머리가 복잡해졌다. 이럴 때일수록 일을 크게 생각하지 말고 잘게 나누고 우선순위를 두어 해야함을 상기했다.  
  
<br>

지치지 않기 위해 조금 더 효율적으로 일을 하고 머리를 써야함을 느낀 하루!
