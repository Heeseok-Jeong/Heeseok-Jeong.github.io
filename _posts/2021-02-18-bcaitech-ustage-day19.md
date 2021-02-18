---
layout : post
title : Ustage Day 19
subtitle : Transformer
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

- [Transformer](#transformer)
- [Multi-head Attention 구현](#multi-head-attention-구현)
- [Masked Multi-head Attention 구현](#masked-multi-head-attention-구현)
- [Byte Pair Encoding](#byte-pair-encoding)
- [피어 세션](#피어-세션)
- [Today I Felt](#today-i-felt)

<br>

<hr>

<br>

# Transformer

<br>

## 참고

지난 번에 배운 Transformer 내용도 참고하자.

- [https://heeseok-jeong.github.io/2021/02/04/bcaitech-ustage-day14.html](https://heeseok-jeong.github.io/2021/02/04/bcaitech-ustage-day14.html)

![image1]({{ site.baseurl }}/assets/img/ustage_day19/1.png)

## 특징

- RNN 구조 대신 Attention 으로만 번역 모델 구성
- Long-Term Dependency 해결
- 인코더와 디코더에서 어텐션 사용
- 기존 어텐션과 다르게 자기 문장 내에서 어텐션을 수행하므로 셀프 어텐션
- 간단한 내적만 수행하면 자기 단어의 가중치가 높을 것

    → 쿼리, 키, 밸류를 이용한 어텐션 방법으로 해결

- 쿼리 : 해당 단어와 관련된 벡터
- 키 : 문장 내 각 단어와 관련된 벡터
- 밸류 : 쿼리와 키로 구한 확률분포와 결합되는 벡터
- 한 단어의 쿼리와 모든 단어의 키를 내적하여 벡터를 만듦 → 이를 소프트맥스하여 가중치를 만들고 모든 단어에 대한 밸류 벡터의 가중 평균을 얻어냄
- 멀리 있는 단어끼리도 정보를 알 수 있음

![image2]({{ site.baseurl }}/assets/img/ustage_day19/2.png)

## Scaled Dot-Product Attention

- 입력 : 한 단어의 Q 와 모든 단어의 K, V (Q, K, V, Output 은 모두 벡터임)
- 출력은 밸류 벡터의 가중합임
- Q, K 는 내적을 하기 때문에 차원이 같아야 함. V 는 상관없음 (실제 구현에서는 Q, K, V 차원 같게 함)
- 수식

    ![image3]({{ site.baseurl }}/assets/img/ustage_day19/3.png)

    - scaled (루트 k차원으로 나누기)
        - dk 가 커지면 분산도 커짐 → 소프트맥스 했을 때 특정 값이 매우 크게 나오는 문제 발생
        - 이를 해결하기 위해 scaled 수행

    ![image4]({{ site.baseurl }}/assets/img/ustage_day19/4.png)

## Multi-head Attention

- 단일 어텐션은 단어들끼리 연관성을 짓는 방법이 하나이므로 만약 잘못된 연관이 지어질 경우 성능이 낮아짐

    ![image5]({{ site.baseurl }}/assets/img/ustage_day19/5.png)

- 멀티 헤드 어텐션 수행 후 concat 하고 다시 원래 차원으로 돌려냄
- Cost

    ![image6]({{ site.baseurl }}/assets/img/ustage_day19/6.png)

    - Complexity per Layer : RNN 에 비해 Self-Attention 은 각 어텐션에 대한 정보를 지녀야하므로 더 많은 메모리 소요
    - Sequential Operations : 하지만 GPU 개수만 된다면 병렬 연산이 가능하므로 RNN 에 비해 좋음 (RNN 은 순차진행이니까 병렬해도 오래걸림)
        - 따라서 트랜스포머는 RNN 에 비해 메모리는 많이 먹지만 학습 시간은 적게 듦
    - Sequential Operations : 멀리 있더라도 뒤에 있는 단어가 앞 단어를 참조할 수 있음

## Block 단위로 보기

![image7]({{ site.baseurl }}/assets/img/ustage_day19/7.png)

- Multi-Head Attention 부분과 Feed Forward 부분으로 나뉨
- 각 부분은 만들어진 벡터에 Residual connection (이를 위해서는 입력과 출력 벡터 차원이 같아야 함) 을 진행하고 Norm 을 수행

### Layer Normalization

- Normalization
    - 딥러닝에 다양한 Normalization 존재 → 주어진 다수의 sample 에 대해 평균을 0, 분산을 1 로 만든 뒤 원하는 평균과 분산으로 구성할 수 있도록 함
    - Normalization 을 거친 후 y = 2x + 3 의 x 에 넣어주면 (affine transformation) , 평균은 3, 분산은 2 의 제곱이 됨
    - 이들은 경사하강법의 파라미터가 됨

        ![image8]({{ site.baseurl }}/assets/img/ustage_day19/8.png)

- Layer Normalization
    - 각 레이어에 대해 평균과 표준편차를 구해서 normalization 수행, 이후 affine transformation 수행

        ![image9]({{ site.baseurl }}/assets/img/ustage_day19/9.png)

    - 학습 안정화 + 성능 조금 더 끌어올림

## Positional Encoding

- RNN 과 다르게 Self-attention 기반 모듈은 가중치를 구하면 순서에 상관없어짐. 마치 순서를 고려하지 않는 집합으로 인코딩하는 것과 같아짐
- 따라서 위치 정보를 주는 Positional Encoding 필요
- I go home 에서 I 가 처음이라는 것을 벡터에 기록해줌
    - 간단한 방법으로는 I = [3, -2, 4] 에서 첫 번째에 1000 을 더해줌 [1003, -2, 4]
    - 이런 식으로 유니크하게 순서를 알 수 있게 특정 상수를 벡터에 더해줌. sin, cos 주기 함수 사용

    ![image10]({{ site.baseurl }}/assets/img/ustage_day19/10.png)

    - dim 개수만큼 특정한 sin, cos 그래프가 생김
    - 어떤 단어가 어떤 위치에 있었는지 알 수 있게 됨

## Warm-up Learning Rate Scheduler

- 학습 중에 러닝 레이트를 적절히 변경시킴

    ![image11]({{ site.baseurl }}/assets/img/ustage_day19/11.png)

## 인코딩 과정

- 단어 임베딩
- 포지셔널 인코딩 더하기
- 멀티헤드 어텐션 수행
- Residual (원래값 더하기) 후 Layer Normalization
- Feed Forward 수행
- Residual (원래값 더하기) 후 Layer Normalization
- 위 과정을 N 번 (6, 12, 24 등, 독립적인 파라미터 가짐) 만큼 진행. stack

![image12]({{ site.baseurl }}/assets/img/ustage_day19/12.png)

- 사진과 같이 멀티헤드 어텐션으로 인해 한 단어가 문장 내 다른 단어와 어떻게 연관짓는지 알 수 있음

## Decoder

![image13]({{ site.baseurl }}/assets/img/ustage_day19/13.png)

- 출력 문장에 대해 임베딩하고 포지셔널 인코딩한 후 Masked Multi-Head Attention 수행하여 Query 만들어냄. 추가적으로 Residual + Norm 수행
- 인코더에서 나온 최종 벡터가 K, V 로 사용되어 Multi-Head Attention (Encoder-Decoder Attention) 수행. 추가적으로 Residual + Norm 수행. 여기서 Residual 덕에 디코더의 문장 정보를 지니게 됨.

    → 인코더의 정보와 디코더의 문장 정보를 잘 결합

- 마지막으로 Feed Forward 수행

## Masked Self-Attention

- 출력 문장에서 이전 단어들에 대해서만 어텐션을 수행. 뒤에 단어들은 masked 가려버림.
- QK 하고 softmax 한 후 뒤에 해당되는 부분을 0 으로 만듦. 이후 row 별로 합이 1 이 되도록 다시 softmax

![image14]({{ site.baseurl }}/assets/img/ustage_day19/14.png)

## 결과

![image15]({{ site.baseurl }}/assets/img/ustage_day19/15.png)

**Further Reading**

- [Attention is all you need, NeurIPS'17](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Group Normalization](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)

**Further Question**

- Attention은 이름 그대로 어떤 단어의 정보를 얼마나 가져올 지 알려주는 직관적인 방법처럼 보입니다. Attention을 모델의 Output을 설명하는 데에 활용할 수 있을까요?
    - 참고: [Attention is not explanation](https://arxiv.org/pdf/1902.10186.pdf)
    - 참고: [Attention is not not explanation](https://www.aclweb.org/anthology/D19-1002.pdf)

<br>

<hr>

<br>

# Multi-head Attention 구현

<br>

## 목적

- 이미 트랜스포머 모델이나 멀티헤드 어텐션은 딥러닝 프레임워크에서 잘 구현돼있지만 간략하게나마 구조를 직접 이해해보는 것

## 과정

### 필요 패키지 import

### 데이터 전처리

- 최대 길이 기준으로 데이터 패딩

### Hyperparameter 세팅 및 Embedding

- n_head = 8, d_model = 512 (d_model 은 n_head 로 나눠 떨어져야함)
- 임베딩 생성 및 배치 임베딩

### Linear Transformation & 여러 head 로 나누기

- w_q, w_k, w_v 생성
- q, k, v 생성

    ```python
    q = w_q(batch_emb)  # (B, L, d_model)
    k = w_k(batch_emb)  # (B, L, d_model)
    v = w_v(batch_emb)  # (B, L, d_model)

    print(q.shape)
    print(k.shape)
    print(v.shape)
    ```

- 멀티 헤드 어텐션으로 쪼개기, 차원을 n_head 개로 쪼갬

    ```python
    batch_size = q.shape[0]
    d_k = d_model // num_heads

    q = q.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
    k = k.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
    v = v.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)

    print(q.shape)
    print(k.shape)
    print(v.shape)
    ```

### Scaled dot-product self-attention 구현

- 수식대로 attn_values 를 구함

    ```python
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, L, L)
    attn_dists = F.softmax(attn_scores, dim=-1)  # (B, num_heads, L, L)

    print(attn_dists)
    print(attn_dists.shape)

    attn_values = torch.matmul(attn_dists, v)  # (B, num_heads, L, d_k)

    print(attn_values.shape)
    ```

### 각 head 의 결과물 병합

- 다시 멀티 헤드를 하나로 합쳐야함

    ```python
    attn_values = attn_values.transpose(1, 2)  # (B, L, num_heads, d_k)
    attn_values = attn_values.contiguous().view(batch_size, -1, d_model)  # (B, L, d_model)

    print(attn_values.shape)
    ```

### 전체 코드

```python
class MultiheadAttention(nn.Module):
  def __init__(self):
    super(MultiheadAttention, self).__init__()

    # Q, K, V learnable matrices
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    # Linear transformation for concatenated outputs
    self.w_0 = nn.Linear(d_model, d_model)

  def forward(self, q, k, v):
    batch_size = q.shape[0]

    q = self.w_q(q)  # (B, L, d_model)
    k = self.w_k(k)  # (B, L, d_model)
    v = self.w_v(v)  # (B, L, d_model)

    q = q.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
    k = k.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
    v = v.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)

    q = q.transpose(1, 2)  # (B, num_heads, L, d_k)
    k = k.transpose(1, 2)  # (B, num_heads, L, d_k)
    v = v.transpose(1, 2)  # (B, num_heads, L, d_k)

    attn_values = self.self_attention(q, k, v)  # (B, num_heads, L, d_k)
    attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, d_model)  # (B, L, num_heads, d_k) => (B, L, d_model)

    return self.w_0(attn_values)

  def self_attention(self, q, k, v):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, L, L)
    attn_dists = F.softmax(attn_scores, dim=-1)  # (B, num_heads, L, L)

    attn_values = torch.matmul(attn_dists, v)  # (B, num_heads, L, d_k)

    return attn_values
```

<br>

<hr>

<br>

# Masked Multi-head Attention 구현

<br>

## 과정

### 필요 패키지 import

### 데이터 전처리

### Hyperparameter 세팅 및 embedding

### Mask 구축

- 자기 번호보다 뒷부분과 패딩인 부분을 모두 False 로 두어 mask 설정 (정렬돼있으므로 패딩 발견되면 밑에는 가려도 됨). 나머지는 attention 해도 되므로 True 로 설정

    ```python
    # 패딩 부분 False
    padding_mask = (batch != pad_id).unsqueeze(1)  # (B, 1, L)

    print(padding_mask)
    print(padding_mask.shape)

    # 자기 번호 뒷부분 False, tril 함수 지원
    nopeak_mask = torch.ones([1, max_len, max_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L)

    print(nopeak_mask)
    print(nopeak_mask.shape)

    # 합치기
    mask = padding_mask & nopeak_mask  # (B, L, L)

    print(mask)
    print(mask.shape)
    ```

### Linear Transformation & 여러 head 로 나누기

- 위와 동일

### Masking 이 적용된 self-attention 구현

- False 부분은 -무한대로 만듦 → softmax 에서 0 으로 바뀜

    ```python
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, L, L)

    masks = mask.unsqueeze(1)  # (B, 1, L, L)
    masked_attn_scores = attn_scores.masked_fill_(masks == False, -1 * inf)  # (B, num_heads, L, L)

    print(masked_attn_scores)
    print(masked_attn_scores.shape)

    attn_dists = F.softmax(masked_attn_scores, dim=-1)  # (B, num_heads, L, L)

    print(attn_dists)
    print(attn_dists.shape)

    attn_values = torch.matmul(attn_dists, v)  # (B, num_heads, L, d_k)

    print(attn_values.shape)
    ```

### 전체 코드

```python
class MultiheadAttention(nn.Module):
  def __init__(self):
    super(MultiheadAttention, self).__init__()

    # Q, K, V learnable matrices
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    # Linear transformation for concatenated outputs
    self.w_0 = nn.Linear(d_model, d_model)

  def forward(self, q, k, v, mask=None):
    batch_size = q.shape[0]

    q = self.w_q(q)  # (B, L, d_model)
    k = self.w_k(k)  # (B, L, d_model)
    v = self.w_v(v)  # (B, L, d_model)

    q = q.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
    k = k.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
    v = v.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)

    q = q.transpose(1, 2)  # (B, num_heads, L, d_k)
    k = k.transpose(1, 2)  # (B, num_heads, L, d_k)
    v = v.transpose(1, 2)  # (B, num_heads, L, d_k)

    attn_values = self.self_attention(q, k, v, mask=mask)  # (B, num_heads, L, d_k)
    attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, d_model)  # (B, L, num_heads, d_k) => (B, L, d_model)

    return self.w_0(attn_values)

  def self_attention(self, q, k, v, mask=None):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, L, L)

    if mask is not None:
      mask = mask.unsqueeze(1)  # (B, 1, L, L) or  (B, 1, 1, L)
      attn_scores = attn_scores.masked_fill_(mask == False, -1*inf)

    attn_dists = F.softmax(attn_scores, dim=-1)  # (B, num_heads, L, L)

    attn_values = torch.matmul(attn_dists, v)  # (B, num_heads, L, d_k)

    return attn_values

multihead_attn = MultiheadAttention()

outputs = multihead_attn(batch_emb, batch_emb, batch_emb, mask=mask)  # (B, L, d_model)

print(outputs)
print(outputs.shape)
```

### Encoder-Decoder attention

- Q, K, V 만 다를 뿐 나머지는 똑같음

    ```python
    q = w_q(trg_emb)  # (B, T_L, d_model)
    k = w_k(src_emb)  # (B, S_L, d_model)
    v = w_v(src_emb)  # (B, S_L, d_model)

    ...

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, T_L, S_L)
    attn_dists = F.softmax(attn_scores, dim=-1)  # (B, num_heads, T_L, S_L)

    attn_values = torch.matmul(attn_dists, v)  # (B, num_heads, T_L, d_k)

    print(attn_values.shape)
    ```

    - attn_dists 는 trg 단어가 인코더의 단어와의 관계를 파악함

    ![image16]({{ site.baseurl }}/assets/img/ustage_day19/16.png)

<br>

<hr>

<br>

# Byte Pair Encoding

<br>

## Subword 를 만드는 방법으로 BPE 사용

- 학습 과정에서 low, lower, newest, widest  이라는 단어들로 vocab 을 만들 때 해당 단어들만 사용하면 테스트 과정에서 lowest 라는 단어는 vocab 에 없기 때문에 UNK 토큰으로 처리되어 번역 성능이 상당히 낮아짐
- 이를 해결하기 위해 BPE 를 사용
- 방법
    - 단어들을 문자 단위로 쪼개고 두 문자씩 합쳤을 때의 등장 횟수가 가장 많은 문자를 vocab 에 넣는 행위를 반복함
    - 아래는 단어와 출연 횟수를 나타냄
    - low : 5, lower : 2, newest : 6, widest : 3
        - 처음 vocab : [l, o, w, e, r, n, w, s, t, i, d]

            BPE 1 번 수행

            - l o w → (lo), (ow)
            - l o w e r, 2 → (lo), (ow) ... (er)
            - n e w e s t, 6 → (ne), ... (es), (st)
            - w i d e s t, 3 → (wi), ... (es), (st)

        ⇒ es 가 9번 등장으로 가장 많음, 결과 vocab : [l, o, w, e, r, n, w, s, t, i, d, es]

        - 다음 BPE 결과 → vocab : [l, o, w, e, r, n, w, s, t, i, d, es, est]
    - 이 과정을 반복하다 보면 lowest 라는 단어를 cover 할 수 있게됨

<br>

<hr>

<br>

# 피어 세션

## 수업 질문

- [[MJ] Transformer에서 Batch normalization 대신 layer normalization을 사용하는 이유](https://github.com/boostcamp-ai-tech-4/peer-session/issues/73)
- [[MJ] position encoding](https://github.com/boostcamp-ai-tech-4/peer-session/issues/74)

<br>

<br>

<hr>

<br>

# Today I Felt

## 진로 고민

AI 쪽으로 일을 하고 싶은데 취업 vs 대학원, 번역 vs 챗봇 vs 추천시스템 등 여러 고민거리가 머릿속에 있다. 미래를 생각해서 대학원에 가는 것이 좋지 않을까 이런 마음에 많이 드는 요즘, 부족한 영어실력이 아쉬워 아침에 일어나서 영어 단어를 보는게 좋겠다는 생각이 든다. 머지않아 U 스테이지가 끝나면 주제를 선택해야 하는데 아직 저 주제들 중 어떤 주제를 할진 모르지만 더 끌리고 재밌게 할 수 있는 주제를 선택할 수 있도록 고민해야겠다.

<br>
