---
title: 'NumPy로 GPT-2 구현 | 5. 사전 학습 가중치 적용과 Hugging Face 출력 비교'
date: '2026-11-11 05:29:53'
author: 지티오
tags: Numpy2GPT LLM GPT2 Weight Hugging Face logits sampling
categories: 개발
readingStyle: reflection
---

[앞 글](/Numpy2GPT-2/4_transformer_block/)에서 GPT-2의 계산 과정을 연결했다. 이번에는 공개 GPT-2 가중치를 적용하고, Hugging Face(HF) 모델과 출력이 얼마나 같은지 비교했다.

먼저 **어느 함수에 어떤 가중치를 넣는지**, **그 가중치를 원본 파일 어디에서 가져오는지** 정리한다. 이어서 실제 logits와 오차를 확인하고, 같은 가중치인데도 차이가 생긴 이유를 살펴본다.

## 1. 가중치가 필요한 곳

`@`는 행렬 곱셈, `W`는 입력에 곱하는 행렬, `b`는 결과에 더하는 **편향(bias) 벡터**다. `W`와 `b` 모두 학습된 값을 파일에서 가져온다.

### 임베딩 — `forward()`

토큰 ID와 위치를 벡터로 바꾸는 표가 필요하다. `wte`는 토큰 임베딩 표, `wpe`는 위치 임베딩 표다.

```text
입력 벡터 = wte[토큰 ID] + wpe[위치]
```

### LayerNorm — `layer_norm()`

정규화한 벡터에 곱할 `gamma`와 더할 `beta`가 필요하다.

```text
정규화된 x = (x - 평균) / √(분산 + ε)
출력 = gamma × 정규화된 x + beta
```

평균·분산은 입력에서 계산하고, `ε`는 나눗셈을 안정화하는 작은 상수다. **파일에서는 `gamma`, `beta`를 가져와 성분별로 적용한다.**

### Attention — `multi_head_self_attention()`

Q·K·V를 만들 때와 헤드를 합친 결과를 변환할 때 각각 `W`, `b`가 필요하다. `x`는 첫 LayerNorm을 통과한 입력이다.

```text
Q = x @ W_Q + b_Q
K = x @ W_K + b_K
V = x @ W_V + b_V

출력 = 헤드를 합친 결과 @ W_O + b_O
```

어텐션 확률은 입력마다 계산하고, `W`, `b`는 파일에서 가져온다. Q·K·V로 확률을 구하는 과정은 [Attention 글](/Numpy2GPT-2/3_attention/)에서 다뤘다.

### FFN — `feed_forward()`

벡터를 넓히는 `W_1`, `b_1`과 줄이는 `W_2`, `b_2`가 필요하다. `x`는 두 번째 LayerNorm을 통과한 입력이다.

```text
중간값 = GELU(x @ W_1 + b_1)
출력 = 중간값 @ W_2 + b_2
```

GPT-2에서는 `768 → 3072 → 768`차원으로 계산한다. GELU 자체에는 가져올 가중치가 없다.

### 블록 연결 — `forward()`의 블록 반복

위의 LayerNorm·Attention·FFN을 순서대로 실행하고 잔차를 더한다.

```text
h = x + Attention(LayerNorm_1(x))
출력 = h + FFN(LayerNorm_2(h))
```

각 함수의 가중치를 그대로 사용하며, **잔차를 더하는 데는 별도 가중치가 없다.**

### 최종 출력 — `forward()`

마지막 LayerNorm의 `gamma`, `beta`를 적용한 뒤, 입력에서 썼던 `wte`를 다시 사용한다.

```text
마지막 벡터 = LayerNorm_f(마지막 블록의 출력)
logits = 마지막 벡터 @ wte.T
확률 = softmax(logits)
```

`wte.T`는 `wte`의 행과 열을 바꾼 것이다. **출력층은 입력 임베딩을 공유**하며, softmax에는 별도 가중치가 없다.

==blue:logits는 다음에 올 토큰 후보마다 매긴 점수다.== 점수가 높을수록 모델이 더 유력하게 보는 후보이며, softmax를 적용하면 후보별 확률이 된다.

## 2. 원본 가중치 위치 한눈에 보기

가져올 파일은 Hugging Face의 [`openai-community/gpt2` — `model.safetensors`](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors)다. 로컬에는 `weights/gpt2/model.safetensors`로 저장했다.

아래 표에서 **쓰이는 곳 → 우리 코드의 키 → 원본 파일의 위치** 순서로 대응시켰다. `i`는 블록 번호로, 첫 블록은 `0`, 마지막 블록은 `11`이다.

<div class="bpe-comparison-table" role="region" aria-label="GPT-2 가중치 대응표" tabindex="0">

| 쓰이는 곳 | 우리 코드의 키 | 원본 파일에서 가져올 위치 |
| --- | --- | --- |
| 토큰 임베딩 | `wte` | `wte.weight` |
| 위치 임베딩 | `wpe` | `wpe.weight` |
| 첫 LayerNorm의 배율 | `blocks.i.ln_1.gamma` | `h.i.ln_1.weight` |
| 첫 LayerNorm의 편향 | `blocks.i.ln_1.beta` | `h.i.ln_1.bias` |
| Q를 만드는 행렬 | `blocks.i.attn.W_Q` | `h.i.attn.c_attn.weight[:, 0:768]` |
| Q에 더하는 편향 | `blocks.i.attn.b_Q` | `h.i.attn.c_attn.bias[0:768]` |
| K를 만드는 행렬 | `blocks.i.attn.W_K` | `h.i.attn.c_attn.weight[:, 768:1536]` |
| K에 더하는 편향 | `blocks.i.attn.b_K` | `h.i.attn.c_attn.bias[768:1536]` |
| V를 만드는 행렬 | `blocks.i.attn.W_V` | `h.i.attn.c_attn.weight[:, 1536:2304]` |
| V에 더하는 편향 | `blocks.i.attn.b_V` | `h.i.attn.c_attn.bias[1536:2304]` |
| Attention 출력 행렬 | `blocks.i.attn.W_O` | `h.i.attn.c_proj.weight` |
| Attention 출력 편향 | `blocks.i.attn.b_O` | `h.i.attn.c_proj.bias` |
| 두 번째 LayerNorm의 배율 | `blocks.i.ln_2.gamma` | `h.i.ln_2.weight` |
| 두 번째 LayerNorm의 편향 | `blocks.i.ln_2.beta` | `h.i.ln_2.bias` |
| FFN 확장 행렬 | `blocks.i.ffn.W_1` | `h.i.mlp.c_fc.weight` |
| FFN 확장 편향 | `blocks.i.ffn.b_1` | `h.i.mlp.c_fc.bias` |
| FFN 축소 행렬 | `blocks.i.ffn.W_2` | `h.i.mlp.c_proj.weight` |
| FFN 축소 편향 | `blocks.i.ffn.b_2` | `h.i.mlp.c_proj.bias` |
| 마지막 LayerNorm의 배율 | `ln_f.gamma` | `ln_f.weight` |
| 마지막 LayerNorm의 편향 | `ln_f.beta` | `ln_f.bias` |

</div>

**Q·K·V는 원본에서 하나로 붙어 있다.** 그래서 `c_attn.weight`의 열을 앞 768개는 Q, 다음 768개는 K, 마지막 768개는 V로 나눈다. 편향도 같은 순서로 나눈다. `0:768`은 인덱스 `0~767`을 뜻한다.

예를 들어 첫 블록의 `W_Q`는 `h.0.attn.c_attn.weight`의 앞 768개 열을 꺼내 `weights["blocks.0.attn.W_Q"]`에 넣으면 된다.

## 3. 기존 구현에 가중치를 넣고 비교하기

기존 NumPy 구현에 위 가중치를 넣고 HF의 `GPT2LMHeadModel`과 비교했다.

원래 Attention에 빠져 있던 `b_Q`, `b_K`, `b_V`, `b_O`를 반영했다. 예를 들어 Q는 `Q = x @ W_Q + b_Q`로 계산한다.

두 모델에 같은 가중치와 토큰 ID를 넣고 logits의 절대오차를 구했다. `hf_model`은 HF 모델이다.

```python
from notebook_model import forward

numpy_logits = forward(token_ids, weights, config)["logits"]
with torch.inference_mode():
    hf_logits = hf_model(
        torch.tensor(token_ids[None, :]), use_cache=False,
    ).logits[0].numpy()

error = np.abs(numpy_logits - hf_logits)
print(error.max(), error.mean())  # 최대·평균 절대오차
```

**모든 입력 위치에서 다음 토큰 후보 50,257개의 점수를 각각 대조**했다. 최대 절대오차는 그중 가장 큰 차이이고, 평균 절대오차는 전체 차이의 평균이다.

## 4. 테스트에 사용한 텍스트

43개 텍스트를 골랐다. 아래는 종류별 실제 입력 예시다.

| 종류 | 실제 입력 예시 | 입력 수 |
| --- | --- | ---: |
| 영어 | `Hello, my name is` / `The capital of France is` | 5 |
| 한국어 | `안녕하세요. 오늘은 직접 만든 모델의 출력을 비교합니다.` | 5 |
| 다국어·혼합 | `오늘의 topic은 attention입니다. Let us compare the logits carefully.` | 5 |
| 코드·JSON·SQL | `{"name": "GPT-2", "layers": 12, "enabled": true, "weights": [0.1, -0.2]}` | 5 |
| 숫자·수식 | `1 + 2 = 3, 10 - 7 = 3, and 12 * 12 = 144.` | 5 |
| 공백·줄바꿈 | `First line.\n\nSecond paragraph.\nThird line.\n` | 5 |
| 반복 문자열 | `hello ` 32회, `0 ` 128회 반복 등 | 5 |
| 유니코드·이모지 | `😀 😃 🙂 🚀 🧠 🌏 GPT-2!` | 5 |
| 짧은 입력·특수 토큰 | `Hi`, 줄바꿈 하나, `<\|endoftext\|>` | 3 |

표의 `\n`은 실제 줄바꿈 문자다.

여기에 길이 실험 14개를 더했다. 다음 문장으로 시작하는 자연문을 이어 붙인 뒤 앞에서부터 `1, 8, 32, 128, 256, 512, 1024`개 토큰을 사용했다. 별도로 ` hello`에 해당하는 한 토큰도 같은 길이만큼 반복했다.

```text
On Monday, Mira opened the laboratory before sunrise and checked the instruments.
```

==yellow:57개 입력의 토큰 수를 모두 더한 값이 5,361이다.== 즉, 5,361곳에서 다음 토큰을 예측했다.

한 위치마다 다음 토큰 후보 50,257개에 점수를 매기므로, 비교한 logits는 `5,361 × 50,257 = 269,427,777개`다.

실제 입력 전체와 토큰 ID, 입력별 오차는 [비교 결과 파일](./gpt2-logits-comparison.json)에 함께 남겼다.

## 5. logits는 얼마나 달랐을까?

`Hello, my name is`를 입력하면 logits의 크기는 `(5, 50257)`이다. 행 5개는 입력 토큰의 위치, 열 50,257개는 다음 토큰 후보를 나타낸다.

아래는 **`Hello, my name is`를 읽은 뒤 바로 다음에 올 토큰 후보** 중 점수가 높은 다섯 개다. 토큰 앞의 공백도 실제 토큰에 포함된다.

| 다음 토큰 후보 | 내 구현의 logit | HF의 logit | 절대오차 |
| --- | ---: | ---: | ---: |
| `' John'` | -64.44328368 | -64.44327545 | 0.00000823 |
| `' K'` | -64.63306077 | -64.63309479 | 0.00003401 |
| `' J'` | -64.70594887 | -64.70587158 | 0.00007729 |
| `' Michael'` | -64.82553307 | -64.82553864 | 0.00000557 |
| `' L'` | -64.85924954 | -64.85914612 | 0.00010342 |

점수는 음수도 나올 수 있다. 두 모델 모두 ` John`의 점수가 가장 높았고, 그 차이는 약 `0.00000823`이었다.

### 전체 57개 입력의 결과

| 비교 항목 | 결과 |
| --- | ---: |
| 최대 절대오차 | 0.00792682 |
| 평균 절대오차 | 0.00011358 |
| 각 위치의 1위 토큰 일치 | 5,361 / 5,361 |
| 각 위치의 상위 5개 토큰 구성 일치 | 5,361 / 5,361 |
| 허용오차 검사 통과 | 54 / 57 |

평균오차는 전체 logits 기준이다. 상위 5개는 후보의 구성만 비교했으며, 그 안의 순서까지 같다는 뜻은 아니다.

### 가장 큰 차이가 난 위치

가장 큰 차이는 자연문 256토큰 입력의 **228번째 위치**에서 발생했다. 그 위치까지 읽은 뒤, **다음 토큰 후보 ` challeng`(ID `4010`)에 매긴 점수**를 비교한 값이다.

```text
내 구현: -68.972496151672
HF:      -68.980422973633
차이:    0.007926821960
```

256·512·1,024토큰 자연문 입력에서 같은 위치의 차이가 유지됐고, 이 세 입력이 허용오차 검사를 통과하지 못했다. 기준은 각 값에 대해 `오차 ≤ 0.0005 + 0.0001 × |HF logit|`이었다.

즉, **가장 높은 점수를 받은 토큰은 모두 같았지만, logits의 숫자까지 같지는 않았다.**

## 6. 같은 가중치인데 왜 차이가 났을까?

내 코드에서는 Attention 점수를 나누는 부분을 원래대로 유지했다.

```python
head_scores = head_scores / np.sqrt(d_head)
```

현재 사용한 NumPy 2.5.4에서는 `np.sqrt(d_head)`가 float64 값을 만들고, 이 나눗셈부터 중간값도 float64가 된다. HF 모델은 float32로 계산했으므로, **기본 비교에서는 두 모델의 계산 정밀도가 달랐다.**

float32와 float64는 중간값을 저장할 수 있는 정밀도가 달라, 계산 중 반올림 결과도 달라질 수 있다.

<div class="llm-example" role="note" aria-label="float32와 float64의 반올림 차이 예시">

**설명용 예시: 정밀도에 따른 반올림**

```text
계산: (100,000,000 + 1) - 100,000,000

float32 결과: 0
float64 결과: 1
```

float32는 이 크기에서 `+1`의 차이를 저장하지 못해 중간값이 반올림된다.

</div>

모델에서도 작은 차이가 이후 층의 계산에 영향을 줄 수 있다. float32 가중치를 float64로 바꿔도 가중치 값 자체는 유지되고, **이후 계산을 수행하는 정밀도**가 달라진다.

중간 출력을 보니 임베딩은 정확히 같았고, 첫 블록부터 차이가 생겼다. 차이가 컸던 두 입력을 **양쪽 모두 float64로 다시 계산**했다.

| 다시 비교한 입력 | 기존 계산의 최대오차 | 양쪽 모두 float64일 때 |
| --- | ---: | ---: |
| 자연문 256토큰 | 0.00792682 | 약 1.17 × 10⁻¹¹ |
| `0 ` 128회 반복 | 0.00264310 | 약 2.59 × 10⁻¹² |

두 입력 모두 차이가 크게 줄었다. HF 모델 자체도 같은 자연문 입력에서 float32와 float64 결과 사이에 최대 약 `0.00749`의 차이가 있었다.

이 재실험은 **계산 정밀도가 오차의 주요 요인임을 뒷받침한다.** 다만 float64 재실험은 두 입력만 확인한 결과다.

## 7. 샘플링 방식에 따른 출력 비교

앞에서는 두 구현의 logits를 비교했다. 이번에는 **같은 NumPy 모델에서 다음 토큰을 고르는 방법만 바꿔** 문장을 생성했다.

공통 입력은 앞에서 사용한 문장이다.

```text
Hello, my name is
```

생성한 토큰을 입력 뒤에 하나씩 붙여 다시 순전파했다. 각 방식은 **최대 40토큰**을 생성하고, 무작위 선택을 하는 아홉 조건은 각각 **시드 42**로 시작했다. Greedy는 무작위 선택을 하지 않는다.

| 방식 | 비교한 설정 |
| --- | --- |
| `Greedy` | 점수가 가장 높은 토큰 선택 |
| 온도만 조절 | `T=0.1·1.0·2.0`, 후보 제한 없음 |
| `Top-k` | `k=50` 고정, `T=0.1·1.0·2.0` 각각 적용 |
| `Top-p` | `p=0.9` 고정, `T=0.1·1.0·2.0` 각각 적용 |

Temperature는 `softmax(logits / T)`로 적용했다. top-k와 top-p에서도 **먼저 온도를 적용한 뒤 후보를 제한**했다. top-k는 상위 50개, top-p는 확률이 높은 순서로 누적합이 0.9 이상이 되는 최소 후보 집합에서 선택한다. 두 제한은 각각 따로 적용했다.

아래는 입력 문장까지 포함한 실제 출력이다. 열 결과 모두 종료 토큰이 나오기 전에 40토큰 한도에 도달했으므로, 문장 중간에서 끊긴 부분도 그대로 남겼다.

### Greedy와 Temperature

<div class="llm-example" role="note" aria-label="후보 제한 없는 생성 결과">

**실제 생성 결과**

<p><strong>Greedy</strong></p>
<p>Hello, my name is John. I'm a writer, and I'm a writer. I'm a writer. I'm a writer. I'm a writer. I'm a writer. I'm a writer. I'm</p>

<p><strong>Temperature 0.1</strong></p>
<p>Hello, my name is John. I'm a freelance writer and I'm a freelance writer. I'm a freelance writer. I'm a freelance writer. I'm a freelance writer. I'm a freelance writer. I'm</p>

<p><strong>Temperature 1.0</strong></p>
<p>Hello, my name is Timothy Leifi<br />
<br />
Aren't we all too familiar with Phyllis Nylsbury's Oliver Crown satire Letters to the Vain Alarm?, at which she says, "it was</p>

<p><strong>Temperature 2.0</strong></p>
<p>Hello, my name is cis Bull 311 Sylvacosaurus Anat Carrierald Ghining057 commits Casesbb wa waist had 460 Creation555 scope 353 Coral Penguin earthguard 3 base Scarlet Gundam Dickinson Burn poison Bengrist25 makers iT Tow</p>

</div>

### Top-k — k=50

<div class="llm-example" role="note" aria-label="Top-k의 Temperature별 생성 출력">

**실제 생성 결과**

<p><strong>Top-k (k=50, T=0.1)</strong></p>
<p>Hello, my name is John. I'm a freelance writer and I'm a freelance writer. I'm a freelance writer. I'm a freelance writer. I'm a freelance writer. I'm a freelance writer. I'm</p>

<p><strong>Top-k (k=50, T=1.0)</strong></p>
<p>Hello, my name is Peter. What do I wish for now? A life?"<br />
<br />
"I really do wish for one of those kind things. And a really nice home!"<br />
<br />
"I will need to come</p>

<p><strong>Top-k (k=50, T=2.0)</strong></p>
<p>Hello, my name is Thomas "Redemption" Perry . Red and Black "Hotdog!" are all good. Just call my name Tommy Smith . " Red" is how much trouble that would take. I don't care</p>

</div>

### Top-p — p=0.9

<div class="llm-example" role="note" aria-label="Top-p의 Temperature별 생성 출력">

**실제 생성 결과**

<p><strong>Top-p (p=0.9, T=0.1)</strong></p>
<p>Hello, my name is John. I'm a writer and a writer's assistant at the University of California, Berkeley. I'm a writer and a writer's assistant at the University of California, Berkeley. I'm a writer</p>

<p><strong>Top-p (p=0.9, T=1.0)</strong></p>
<p>Hello, my name is Sacha Rodrigues, brother of Lieutenant. I am Commander Zaha Hadid.<br />
<br />
BAUGHTARI [confused]<br />
<br />
Your thoughts...<br />
<br />
So, Areli (</p>

<p><strong>Top-p (p=0.9, T=2.0)</strong></p>
<p>Hello, my name is Brassstrong Buccaneers 121 other tantal 134 invites .Good afford inquireername inspectors2018 popular Queens spe printf obscure050 medium825 6000 rumoursTr recoverchTherenikbeck ENTER Linux 07 chocolate battle 2015 Original ChurchCBS</p>

</div>


출력 원문·생성 토큰 ID·설정은 [샘플링 비교 결과 파일](./gpt2-sampling-comparison.json)에 남겼다.

