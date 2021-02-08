---
layout : post
title : Ustage Day 14
subtitle : RNN|LSTM|Transformer
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

- [RNN 첫걸음](#rnn-첫걸음)
- [Sequential Models - RNN](#sequential-models---rnn)
- [Sequential Models - Transformer](#sequential-models---transformer)
- [피어 세션](#피어-세션)
- [Today I Felt](#today-i-felt)

<br>

<hr>

<br>

# RNN 첫걸음

<br>

## 시퀀스 데이터

### 시퀀스 데이터 이해하기

- 순차적으로 들어오는 데이터, 소리, 문자열, 주가 등
- 시계열 (time-series) 데이터는 시간 순서에 따라 나열된 데이터로 시퀀스 데이터에 속함
- 시퀀스 데이터는 독립동등분포 (i.i.d.) 가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀜
    - 개가 사람을 물었다 → 사람이 개를 물었다 : 순서가 바뀌면 데이터의 확률분포가 바뀜, 의미가 바뀜, 과거의 정보와 앞뒤 맥락으로 의미를 유추하기 때문

### 시퀀스 데이터 다루기

- 이전 시퀀스의 정보를 가지고 앞으로 발생할 데이터의 확률분포를 다루기 위해 조건부확률을 이용할 수 있음

    ![image1]({{ site.baseurl }}/assets/img/ustage_day14/1.png)

    ![image2]({{ site.baseurl }}/assets/img/ustage_day14/2.png)

    - 베이즈 정리 사용
    - 시퀀스 데이터를 분석할 때 모든 과거 정보들이 필요한 것은 아님
        - 삼성 주가를 예측하는데 설립 당시 정보는 필요 없기 때문
        - 문장도 마찬가지로 문장 예측을 하는데 책의 처음부터 알 필요는 없음
- 방법 1
    - 시퀀스 데이터를 다루기 위해서는 길이가 가변적인 데이터를 다룰 수 있는 모델이 필요함
- 방법 2
    - 하지만, 위에서 말한 것 처럼 모든 앞선 내용을 알 필요가 없기 때문에 고정된 길이 $\tau$ 만큼의 시퀀스만 사용하고, 이 경우 **AR**($\tau$) (Autoregressive Model) 자기회귀모델이라고 부름
    - 예시
        - 10 번째 단어를 예측할 때, 타우가 3 이라면 9, 8, 7 단어를 이용해 예측
        - 11 번째면 10, 9, 8 단어로 예측
- 방법 3
    - 바로 이전 정보를 제외한 나머지 정보들을 $H_t$ 라는 잠재변수로 인코딩해서 활용하는 **잠재 AR 모델** 사용
    - 예시
        - 10 번째 단어를 예측할 때, $H_t$ 는 8~1 단어가 되고 이를 이용해 예측
        - 11 번째 단어는 $H_\mathit{t+1}$ 는 9~1 단어가 되고 이를 이용해 예측

        Q. 바로 하나 전 정보 빼는거랑 안빼는거랑, 방법 1 이랑 방법 3이랑 무슨 차이?

        ⇒ 그게 아니라 t-1 번째 단어와, 잠재변수 $H_t$ 두 개를 이용해서 t 번째 단어를 예측하는 것!

        ⇒ RNN

        ![image3]({{ site.baseurl }}/assets/img/ustage_day14/3.png)

<br>

<br>

## Recurrent Neural Network

### RNN 이해하기

- 가장 기본적인 RNN 모형은 MLP 와 유사함

    ![image4]({{ site.baseurl }}/assets/img/ustage_day14/4.png)

    - 이 모델은 t 번째 입력이 들어오기 때문에 과거의 정보를 다룰 수 없음
    - 어떻게 과거의 정보를 H 에 담을까?
- RNN 은 이전 순서의 잠재변수와 현재의 입력을 활용하여 모델링

    ![image5]({{ site.baseurl }}/assets/img/ustage_day14/5.png)

    - 가중치 행렬
        - 입력 데이터를 선형모델을 통해 잠재변수로 인코딩해주는 $W_X^\mathit{(1)}$ 과 이전 시점의 잠재변수로부터 정보를 받아 현재 시점의 잠재변수로 인코딩해주는 $W_H^\mathit{(1)}$가 있음
        - 이렇게 만들어진 잠재변수를 통해 출력으로 만들어주는 $W^\mathit{(2)}$ 가 있음
    - 이 가중치 행렬들은 t 에 따라 변하지 않음! (중요)
    - t 에 따라 변하는 것들은 잠재변수와 입력 데이터!
- RNN 의 **역전파**는 잠재변수의 연결그래프에 따라 순차적으로 계산
    - 즉 계산 그래프의 거꾸로 흐름, 잠재 변수들의 연결 그래프에 따라 순차적으로 진행됨, 맨 마지막 시점의 그래디언트가 타고 타고 올라와서 과거 그래디언트까지 흐름 (BPTT)

    ![image6]({{ site.baseurl }}/assets/img/ustage_day14/6.png)

### BPTT, Backpropagation Through Time

- BPTT 를 통해 RNN 의 가중치행렬의 미분을 계산해보면 아래와 같이 미분의 곱으로 이루어진 항이 계산됨

    ![image7]({{ site.baseurl }}/assets/img/ustage_day14/7.png)

    - 최종적으로 나오는 프로덕트텀 (i+1 부터 t 까지 잠재변수 곱) 을 보면 시퀀스 길이가 길어질수록 곱해지는 텀들이 불안정해짐, 이 값이 1 보다 크면 미분값이 매우 커지고 1보다 작으면 미분값이 매우 작아지기 때문 (기울기 소실)
    - 일반적인 BPTT 를 모든 시점에 적용하면 학습이 어려움

### 기울기 소실의 해결책?

- 시퀀스 길이가 길어지는 경우 BPTT 를 통한 역전파 알고리즘의 계산이 불안정해지므로 길이를 끊어줘야 함
⇒ **truncated BPTT**
    - BPTT 를 모든 시점에 적용하지 않고 블록으로 끊어서 수행

    ![image8]({{ site.baseurl }}/assets/img/ustage_day14/8.png)

    - 잠재변수들에 들어오는 그래디언트를 보면 미래에서 $H_\mathit{t+1}$ 까지 들어오는 그래디언트는 받지만, $H_t$ 에는 전달하지 않음, $H_t$ 는 $O_t$ 에서 들어오는 그래디언트르 받음
- 하지만 완전한 해결책이 되지 않음, 따라서 길이가 긴 시퀀스를 처리하기 힘든 Vanilla RNN 대신 LSTM 이나 GRU 등장

<br>

<hr>

<br>

# Sequential Models - RNN

<br>

## Sequential Model

- 비디오, 오디오, 말 등은 모두 시퀀셜 데이터
- 시퀀셜 데이터를 처리하는데 가장 큰 어려움
    - 얻고 싶은 것은 하나의 라벨임, 내가 하는 말이 무엇인지
    - 그런데 말이 몇 단어가 될지 알 수가 없음, 따라서 CNN 을 사용할 수가 없음
    - 몇 개의 입력이 들어오든 상관없이 모델이 동작해야함

### Naive sequence model

- 이전 데이터로 다음 단어를 예측해보자
- 고려해야하는 정보량이 늘어남

    ![image9]({{ site.baseurl }}/assets/img/ustage_day14/9.png)

### Autoregressive model

- Fix the past timespan, 과거 내용 중 볼 내용 크기를 고정

    ![image10]({{ site.baseurl }}/assets/img/ustage_day14/10.png)

### Markov model (first-order autoregressive model)

- Markovian assumption, property 을 가짐
- 내가 가정하기에 현재는 과거 (바로 전 과거) 에만 의존됨
- 내일이 수능이면 오늘 공부만 영향주는가? X → 과거 공부한게 다 누적됨
- 장점 : joint distribution 을 표현하기 쉬움

    ![image11]({{ site.baseurl }}/assets/img/ustage_day14/11.png)

### Latent autoregressive model

- 과거에 많은 정보를 고려해야하는데 할 수가 없음
- 중간에 히든 스테이트 (잠재변수, 과거를 요약) 가 들어감
- 아웃풋은 하나의 과거 정보만 의존하지만 그 이전 잠재변수 (Latent state) 에도 영향받음

    ![image12]({{ site.baseurl }}/assets/img/ustage_day14/12.png)

## Recurrent Nueral Network

- 앞서 말한 구조들을 가장 잘 표현

    ![image13]({{ site.baseurl }}/assets/img/ustage_day14/13.png)

- RNN 은 시간순으로 푼다고 표현, t 에서 t-1 의 정보들도 봄
- 사실 RNN 을 시간순으로 풀면 입력이 굉장히 많은 Fullly-connected layer 와 동일해짐
- 단점, short-term dependencies
    - 과거에 얻어진 어떤 정보들이 요약돼서 미래에서 고려해야 하는데, RNN 은 이 정보들을 같은 방식으로 계속 취합하기 때문에 먼 과거의 정보가 미래에 반영되기 힘듦
    - 메멘토처럼 기억력이 일정 시간이라고 생각하면 됨 → 제한된 사고력

    ![image14]({{ site.baseurl }}/assets/img/ustage_day14/14.png)

    - 더불어 활성함수까지 합쳐지면 더 예전 정보는 영향력이 없어짐, 시그모이드면 Vanishing Gradient, 렐루면 너무 커져서 exponential (폭발적이게) 됨

    ⇒ Long short-term dependencies (LSTM) 로 단점을 극복

- 지금까지 말한 기본 RNN = Vanilla RNN

    ![image15]({{ site.baseurl }}/assets/img/ustage_day14/15.png)

### LSTM

![image16]({{ site.baseurl }}/assets/img/ustage_day14/16.png)

- 어떻게 LSTM 이 롱텀 디펜던시를 해결하는지 볼 것

    ![image17]({{ site.baseurl }}/assets/img/ustage_day14/17.png)

    - input : 단어에 대한 벡터 (총 단어 5 만개면 5 만짜리 one-hot vector)
    - output (hidden state) : 다음번 분포, 단어의 확률을 찾게 해줌 (유일하게 밖으로 나감)
    - previous cell state : 지금까지 들어왔던 t-1 개의 정보를 요약한 정보 (밖으로 나가지 않음)
    - previous hidden state : 전 단계의 아웃풋
- LSTM 의 가장 큰 아이디어는 중간을 흘러가는 cell state! 컨베이너 벨트와 같음. 이 정보를 잘 조작해서 어떤 정보가 유용하고 유용하지 않은지를 다음에 알려줌 ⇒ 게이트 사용
- LSTM 은 게이트로 이해하는게 좋음. 총 3개의 게이트 존재
    - 히든으로부터 받은 정보를 조작해서 cell state 에 영향을 줌

    ![image18]({{ site.baseurl }}/assets/img/ustage_day14/18.png)

    - Forget gate (버릴거 정하기)
        - Decide which information to throw away
        - $f_t$ : 이전 셀스테이트에서 넘어온 정보 중 버릴 것을 정함

        → 이전 히든 정보와 현재 입력을 통해 지울거 정함 

    - Input gate (남길거 정하기)
        - Decide which information to store in the cell state
        - $i_t$ : 이전 히든과 현재 정보를 통해 어떤 정보를 올릴지에 대한 정보
        - $C_t$ (cell state candidate) : 이전 히든과 현재 정보로 tanh 통과해서 모든 값이 정규화된 C 틸다, 셀 스테이트의 예비군
        - 이 두개를 잘 섞으면 새로운 셀 스테이트로 업데이트됨

        → 이전 요약 정보와 현재 입력을 통해 남길거 정함

    ![image19]({{ site.baseurl }}/assets/img/ustage_day14/19.png)

    - Update cell (작업 실시)
        - Update the cell state
        - f 를 통해 버릴거 버리고, C틸다를 통해 어느 값을 올릴지 정함

        → 버릴 것과 남길 것을 토대로 셀 스테이트를 업데이트

    - Output gate
        - Make output using the updated cell state
        - 업데이트된 값을 얼만큼 밖으로 내보낼지 정함

        → 업데이트 된 값을 한 번 더 조작해서 밖으로 빼냄

### GRU, Gated Recurrent Unit

![image20]({{ site.baseurl }}/assets/img/ustage_day14/20.png)

- LSTM 과 다르게 Gate 2 개 (reset gate (폴겟 게이트와 비슷), update gate)
- cell state 없음, hidden state 만 존재
- 똑같은 랭귀지 모델에 대해서 LSTM 보다 GRU 썼을 때 더 성능 좋은 경우가 꽤 있음
    - 파라미터가 적어지기 때문에 퍼포먼스 증가
- 그러나 요즘엔 트랜스포머 때문에 LSTM, GRU 둘 다 대치되버림

## LSTM 실습

- 시퀀셜 데이터 처리하는게 쉽지 않음
    - 단어로 딕셔너리 만들고, 딕셔너리로 원핫벡터 만들고, word2vec 등으로 임베딩 만들어야 함
- 우선 실습은 mnist 사용해서 분류하자
    - 28x28 이니까 row 를 하나의 벡터로 보고 28 디멘젼짜리가 28번 들어왔다고 생각하고 RNN 에 넣기
    - 28번 시퀀스 후에 나오는 어떤 벡터를 하나의 뉴럴 네트워크 더 통과해서 분류할 것
    - 사실 28개짜리 구조 아니여도 상관 없음, 편의상 28로 함
- RNN 이나 LSTM 은 텐서플로우로 구현하기 힘듦, 파이토치는 편함
- 과정
    - 모델 정의
        - init
            - xdim : RNN 입력에 들어가는 하나의 벡터의 크기, 28
            - hdim : RNN 의 cell state 차원과 아웃풋 차원, 256
            - num_layers : RNN 을 위로 몇 단 쌓을지, 3
            - batch_first : 아웃풋으로 나오는게 어떤식으로 나올지 결정, True (False 하면 에러)
        - forward
            - LSTM 은 셀스테이트, 히든 스테이트를 받아야 함 → c0, h0 세팅
            - rnn 에 인풋 x 와 c0, h0 를 넣어 결과 얻음 (x : N x L x Q ⇒ rnn_out : N x L x D)
            - 결과 사이즈를 바꿔줌 (N x K)


    - Check How LSTM Works


        - `N`: number of batches
        - `L`: sequence lengh
        - `Q`: input dim
        - `K`: number of layers
        - `D`: LSTM feature dimension

        `Y,(hn,cn) = LSTM(X)`

        - `X`: [N x L x Q] - `N` input sequnce of length `L` with `Q` dim.
        - `Y`: [N x L x D] - `N` output sequnce of length `L` with `D` feature dim.
        - `hn`: [K x N x D] - `K` (per each layer) of `N` final hidden state with `D` feature dim.
        - `cn`: [K x N x D] - `K` (per each layer) of `N` final hidden state with `D` cell dim.
    - 파라미터 82만개, 엄청 많음
        - 각 과정에 있는 게이트들은 dense layer 이므로 많이 소요
- 결과
    - 이전 정보를 잘 취합해서 마지막에 분류가 된다

**Further Question**

- LSTM에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있습니다. 무엇일까요?

    => Gradient Vanishing 문제를 해결하기 위해 CNN 에서 Residual 을 사용한 것 같이, LSTM 도 과거의 정보를 기억하는 방법을 사용

- Pytorch LSTM 클래스에서 3dim 데이터(batch_size, sequence length, num feature), `batch_first` 관련 argument는 중요한 역할을 합니다. `batch_first=True`인 경우는 어떻게 작동이 하게되는걸까요?

    => output 이 배치가 제일 먼저 나오게 됨

<br>

<hr>

<br>

# Sequential Models - Transformer

<br>

## Sequential Model

- 앞서 말했듯 시퀀셜 모델링은 다음과 같은 이유로 힘듦
    - 단어 중간 생략
    - 단어 뒤에 생략
    - 단어 앞에 생략

    ⇒ 이런 문제를 해결하기 위해 트랜스포머의 셀프-어텐션 등장

## Transformer

- 트랜스포머는 어텐션으로만 설계된 최초의 시퀀스 문제 다루는 모델
    - 기존 RNN (재귀적 수행) 이랑 달라짐
- 트랜스포머는 이미지 분류, detection 등에도 사용됨
    - GPT-3, 달리 등은 셀프 어텐션 사용
- 시퀀스-시퀀스 모델, 주어진 문장을 다른 문장으로 만드는 것, 번역이라면 NMT 모델
- RNN 은 단어 개수대로 모델이 돌아갔다면, 트랜스포머는 인코더, 셀프어텐션 부분은 단어를 한 번에 처리함
- 이해해야 할 것
    - N 개의 단어가 어떻게 한 번에 인코더에 처리되는지
    - 인코더와 디코더는 어떤 정보를 주고 받는지
    - 디코더가 어떻게 문장을 생성해내는지 (이번에는 별로 안 다룸)

### Encoder

- 문장 벡터가 한 번에 들어감
- 인코더는 Self-Attention → Feed Forward Neural Network 로 구성
- 인코더와 디코더에 사용되는 **Self-Attention** 이 트랜스포머의 핵심

    ![image21]({{ site.baseurl }}/assets/img/ustage_day14/21.png)

- 세 개의 단어 (각각 벡터) 가 주어지면 셀프어텐션은  $x_i$ 벡터가 $z_i$ 벡터로 바꿀 때 각각의 단어 x 들을 모두 사용함 → 디펜던시가 있음 (다른 단어들 보는 것)

    ![image22]({{ site.baseurl }}/assets/img/ustage_day14/22.png)

- 피드 포워드는 그냥 원래 MLP 처럼 진행

    ![image23]({{ site.baseurl }}/assets/img/ustage_day14/23.png)

### Self-Attention (이 과정 반드시 이해)

- The animal didn't cross the street because it was too tired. 라는 문장에서 it 이 뭘가리키는지 다른 모든 단어들과의 관계성을 따져야함 → The animal 선택
- Thinking 이라는 단어가 주어졌을 때 인코딩하기 위해 세 개의 벡터 **Queries, Key, Values** 벡터가 사용됨
    - 한 단어마다 Q, K, V 가 생성됨

    ![image24]({{ site.baseurl }}/assets/img/ustage_day14/24.png)

- Thinking 과 Machines 라는 단어들을 인코딩하면 각 단어마다 만들어진 Q, K, V 를 통해 Score 를 생성 → i 번째 단어가 j 번째 단어와 얼마나 유사도가 있는지 qi, kj 를 내적

    ![image25]({{ site.baseurl }}/assets/img/ustage_day14/25.png)

    - i 번째 단어와 나머지 단어들과의 관계를 계산 → Attention

    ![image26]({{ site.baseurl }}/assets/img/ustage_day14/26.png)

- 스코어가 나오면 스코어에 대해노말라이제이션 (루트dk (키벡터차원) = 8로 나눠줌, 값이 너무 커지지 않게 하기 위해).
- 이후 소프트맥스함 → i, j 단어와의 관계를 확률로 표현
- 이 값을 value 와 곱함, 이후 다 더함 Sum

    ![image27]({{ site.baseurl }}/assets/img/ustage_day14/27.png)

⇒ Value 벡터의 웨이트를 구하는 과정. 각 단어 쿼리와 다른 단어 키 곱하고, 노말라이즈하고, 소프트맥스하고, 밸류곱함

- Q, K 는 차원이 같아야함 (내적), V 는 차원 달라도됨
- 행렬로 보자

    ![image28]({{ site.baseurl }}/assets/img/ustage_day14/28.png)

- 수식

    ![image29]({{ site.baseurl }}/assets/img/ustage_day14/29.png)

⇒ Single-head Attention

### 왜 잘될까?

- 어떤 이미지를 CNN 이나 MLP 로 차원을 바꿀 때 인풋 크기가 픽스되면 출력크기도 픽스됨, 필터나 웨이트가 고정되기 때문
- 트랜스포머는 인풋이나 네트워크가 고정되어있더라도, 인코딩하려는 단어와 옆에 주어진 다른 단어들에 따라 출력이 달라질 수 있음 → 훨씬 많은걸 표현할 수 있음

    ⇒ 많이 표현하기 때문에 더 많은 컴퓨테이션 요구함

    - 단어가 1000 개면 1000x1000 (한 번에 처리해야하고, 비용이 N^2 소모됨) 만들어야함. RNN 은 1000 만 필요함
    - 즉, 시간이나 메모리는 더 먹지만 더 표현을 잘하고 플렉서블한 모델임

## Multi-head Attention

- 싱글 헤드 어텐션을 여러번 함
- 하나의 임베딩된 벡터에 대해 Q, K, V 를 여러개 만듦

    ![image30]({{ site.baseurl }}/assets/img/ustage_day14/30.png)

- 한 단어에 대해 만약 8번 셀프 어텐션하면 8개의 인코딩된 결과가 나옴

    ![image31]({{ site.baseurl }}/assets/img/ustage_day14/31.png)

- 임베딩된 차원과 인코딩된 차원은 항상 같아야 함

![image32]({{ site.baseurl }}/assets/img/ustage_day14/32.png)

- 실제로는 한 단어의 임베딩 벡터 차원이 100 이라면 10 짜리 10 개로 나눠서 어텐션 진행 → 10 헤드 어텐션

### positional encoding

- 모든 단어를 한 번에 인코더에 넣으면 단어들의 순서가 사라짐 → 순서를 기억하게 해줌
- 주어진 입력 값에 어떤 값을 더해줌
- 최근에 사용되는 포지셔널 인코딩 결과

    ![image33]({{ site.baseurl }}/assets/img/ustage_day14/33.png)

- 왜 트랜스포머는 입력 임베딩에 독릭적으로 진행되는지 생각해볼 것

![image34]({{ site.baseurl }}/assets/img/ustage_day14/34.png)

## Decoder

![image35]({{ site.baseurl }}/assets/img/ustage_day14/35.png)

- Encoder 는 주어진 단어를 표현하는 것
- Decoder 는 단어를 생성해내야함
- Encoder 에서 Decoder 로 어떤 정보가 갈까?
    - Key, Value 를 보냄
    - i 번째 단어 어텐션 만들 때 Qi 와 Kj 곱하고 차원루트 나누고 Vi 더했음
    - 인풋에 있는 단어들에 대해 출력하고자 하는 어텐션을 만들려면 인풋 단어들의 K 와 V 벡터가 필요함, 가장 상위 레이어 만듦
- 디코더에서 셀프어텐션 레이어는 마스킹을 함, 이전 단어들만 관여하고 뒤에 단어들에 대해서는 어텐션 하지 않음
- 추정할 때도 마찬가지

### Encoder-Decoder Attention

- 디코더에 들어있는 레이어
- 인코더의 K, V 와 출력 단어의 Q 로 어텐션 진행

## 활용

### Vision Transformer

- 원래는 NMT (번역) 문제에만 사용됐는데 이제는 이미지에도 많이 사용
- VIT 라는 논문에서 사용
- 인코더만 활용
- 원래 NMT 는 단어들의 시퀀스가 있었다면, 이미지를 가공해서 맞춰줌 (포지셔널 인코딩도 필요)

### DALL-E

- 문장만 보고 사진을 생성해냄
- GPT-3 활용

## 실습

- Scaled Dot-Product Attention (SDPA) = 셀프 어텐션, 한 층이기 때문에 디코더 쪽에서 사용

    ⇒ Encoder-Decoder Attention

    ![image36]({{ site.baseurl }}/assets/img/ustage_day14/36.png)

    - 원래라면 K, Q 는 차원 같아야하고 V 는 달라도 되지만 여기서 K, V 는 인코더에서 온거고 Q 는 디코더꺼라 K, V 가 같음
    - SPDA 의 목적은 Q 에 대해 인코더를 찾는게 목적
- Multi Head Attention
    - 어텐션 n_head 만큼 만듦

![image37]({{ site.baseurl }}/assets/img/ustage_day14/37.png)

⇒ 꼭 코딩 따라서 다 쳐볼 것, 이해도 하고

**Further Reading**

- [Illustrated transformer (원글)](http://jalammar.github.io/illustrated-transformer/)
- [Illustrated transformer (번역)](https://nlpinkorean.github.io/illustrated-transformer/)
- [Pytorch official Transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

**Further Questions**

- Pytorch에서 Transformer와 관련된 Class는 어떤 것들이 있을까요?

<br>

<hr>

<br>

# 피어 세션

<br>

## 스몰 토크

- 서폿님과의 미니 심리학 토크 (심리 연구의 여러 허점)

## 수업 질문
- [[펭귄] Adadelta 수식](#https://github.com/boostcamp-ai-tech-4/peer-session/issues/55)
- [[펭귄] LSTM과 GRU 비교](#https://github.com/boostcamp-ai-tech-4/peer-session/issues/56)

## 미니 데이터셋 만들기 회의

- 내일까지 참가자 지원
- 주말에 데이터 수집 및 분류
- 각자 학습

<br>

<hr>

<br>

# Today I Felt

<br>

## 다른 짓...

저녁 5시쯤 친구가 던져준 알고리즘 문제를 밤에 풀었어야 했는데 빨리 풀고 남은 강의 봐야지 하며 문제에 달려들었다. 생각보다 안풀려서 시간을 1시간 넘게 써버렸다.. 이게 화근이었다 ㅠ 문제 풀다가 정말 기대했던 부스트캠퍼 3기 선배 Meet Up 도 까먹어버리고.. 흑흑.. 그렇게 예상보다 강의를 늦게 다보게 되며 밤에서야 복습을 끝낼 수 있었다. 딴짓은 할거 다하고 하자..

## 트랜스포머!!!

캡스톤 프로젝트에 사용했던 트랜스포머 모델을 배울 수 있어서 너무 좋았다. 정말 최성준 교수님의 강의는 너무나 만족스럽다. 프로젝트 당시 이해도 못하고 사용했던 트랜스포머라 항상 마음에 걸렸는데 이제는 이해했기 때문에 마음이 한결 가볍다. 앞으로 다양한 모델과 기술을 볼 때는 교수님께서 알려주시지 않기 때문에 스스로 구글링을 잘하고 논문을 보며 이해하는 연습을 해야겠다고 느꼈다.
