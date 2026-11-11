---
title: 'NumPy로 GPT-2 구현 | 6. 구성 요소 제거 실험과 SFT·RLHF·DPO 손실 이해'
date: '2026-11-11 08:09:20'
author: 지티오
tags: Numpy2GPT LLM GPT2 Attention logits 실험 Loss SFT RLHF PPO DPO
categories: 개발
readingStyle: reflection
---

[앞 글](/Numpy2GPT-2/5_apply_weight/)에서 공개 가중치를 적용하고 HF와 logits를 비교했다. 이번에는 **√dₖ 스케일링, causal mask, 잔차 연결을 하나씩 제거**해 출력이 어떻게 달라지는지 확인했다.

입력은 `Hello! My Name Is`다. 가중치는 그대로 두고, 실험마다 한 요소만 12개 블록 전체에서 제거했다.

앞부분에서는 구성 요소를 제거했을 때의 출력을 비교하고, 뒷부분에서는 그 출력으로 다음 토큰 예측 손실과 선호 학습의 손실을 계산한다.

<div role="navigation" aria-label="글 목차">

**글의 흐름**

- **구성 요소 제거 실험:** [실험 조건](#실험-조건과-정상-출력) · [스케일링](#1-dₖ-스케일링-제거) · [causal mask](#2-causal-mask-제거) · [잔차 연결](#3-잔차-연결-제거)
- **손실과 선호 학습:** [다음 토큰 예측](#4-다음-토큰-예측-손실) · [SFT](#5-sft-모범-답변을-따라-배우기) · [RLHF](#6-rlhf-사람이-선호한-답변을-더-자주-만들기) · [DPO](#7-dpo-두-답변의-선호를-직접-배우기)
- [마치며](#마치며)

</div>

## 실험 조건과 정상 출력

GPT-2 small과 같은 토크나이저를 사용했다. 입력의 대소문자와 느낌표를 유지하고, 끝에 공백이나 특수 토큰은 붙이지 않았다.

```text
입력: Hello! My Name Is
토큰: ['Hello', '!', ' My', ' Name', ' Is']
ID:   [15496, 0, 2011, 6530, 1148]
```

토큰 앞의 공백도 토큰의 일부다. logits는 입력 5개 위치마다 후보 50,257개의 점수를 담은 `(5, 50257)` 배열이다.

이 중 **마지막 행** `logits[-1]`, 즉 ` Is` 다음에 올 토큰의 점수를 비교했다.

- **계산 정밀도:** 모든 조건은 float64로 계산했다.
- **가중치:** 공개 float32 값을 변환해 사용했다.
- **실행 조건:** 재학습·dropout·KV cache 없이 실행했다.

정상 구현과 HF의 전체 logits 최대 절대오차는 약 `3.69 × 10⁻¹³`이었다.

<div class="bpe-comparison-table" role="region" aria-label="구성 요소 제거 실험 요약" tabindex="0">

| 조건 | 다음 토큰 1위 | 해당 토큰 확률 | 정상 대비 logits 최대 절대차 |
| --- | --- | ---: | ---: |
| 정상 | `,` | 1.37% | 0.0000 |
| √dₖ 스케일링 제거 | ` My` | 12.48% | 12.6511 |
| causal mask 제거 | ` My` | 51.86% | 19.8112 |
| 잔차 연결 제거 | `\n` | 26.30% | 84.3594 |

</div>

최대 절대차는 **같은 후보 토큰의 점수 차이 중 가장 큰 값**이다. 확률은 마지막 행에 softmax를 적용한 값이며, `\n`은 줄바꿈 토큰이다.

logits는 확률이 아니므로 음수도 가능하다. 모든 점수가 같은 만큼 움직이면 softmax는 같으므로, **점수 차이와 함께 후보 순위·확률의 변화**를 봐야 한다.

## 1. √dₖ 스케일링 제거

### 바뀌는 코드

`multi_head_self_attention()`에서 점수를 나누는 줄을 주석 처리한다. `d_head`는 한 헤드의 차원 수인 `dₖ`다. 여기서는 `768 / 12 = 64`이므로 원래 점수를 `√64 = 8`로 나눈다.

<details>
<summary>스케일링 변경 전·후 코드 보기</summary>

**변경 전**

```python
head_scores = Q_heads @ K_heads.swapaxes(-1, -2)
head_scores = head_scores / np.sqrt(d_head)
```

**변경 후**

```python
head_scores = Q_heads @ K_heads.swapaxes(-1, -2)
# head_scores = head_scores / np.sqrt(d_head)
```

</details>

causal mask와 softmax는 그대로 적용한다.

### logits에서 보인 차이

각 실험의 표는 **제거 후 상위 3개 후보**를 정상 모델의 같은 토큰과 비교한 것이다. logits는 소수 넷째 자리, 확률은 백분율 소수 둘째 자리까지 반올림했다.

<div class="bpe-comparison-table" role="region" aria-label="스케일링 제거 전후 logits" tabindex="0">

| 토큰 | 정상 logits | 스케일링 제거 logits | 제거 후 확률 |
| --- | ---: | ---: | ---: |
| ` My` | -65.2963 | -69.7217 | 12.48% |
| ` Your` | -66.5362 | -69.8778 | 10.68% |
| ` the` | -67.6006 | -70.9743 | 3.57% |

</div>

1위가 `,`에서 ` My`로 바뀌었고, 최대 절대차는 `12.6511`이었다. ` My`의 점수 자체는 낮아졌지만, ==yellow:다른 후보와의 상대적인 점수 관계==가 달라져 1위가 됐다.

첫 블록에서는 각 어텐션 행의 최대 확률을 평균 낸 값이 `71.55% → 95.14%`로 커졌다. 12개 헤드·5개 위치를 평균한 값으로, **특정 입력 위치를 참고하는 비율**이 더 커졌다는 뜻이다. 다음 토큰을 고르는 확률과는 다르다.

### 스케일링의 역할과 탐구 질문

**질문: Q와 K의 내적을 왜 √dₖ로 나눌까?**

나누지 않으면 점수가 한쪽에 몰려 softmax가 하나에 너무 높게 확률을 몰아준다. ==blue:√dₖ로 나누는 것은 어텐션이 한쪽에 지나치게 쏠리지 않도록 점수 크기를 조절하는 과정이다.== 학습에서는 softmax 포화로 기울기가 작아지는 현상도 완화한다.

## 2. causal mask 제거

### 바뀌는 코드

같은 함수에서 미래 위치를 `-np.inf`로 가리는 부분을 주석 처리하고, 점수를 그대로 softmax에 넣는다.

<details>
<summary>causal mask 변경 전·후 코드 보기</summary>

**변경 전**

```python
future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
head_masked_scores = np.where(future_mask, -np.inf, head_scores)
head_weights = softmax(head_masked_scores)
```

**변경 후**

```python
# future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
# head_masked_scores = np.where(future_mask, -np.inf, head_scores)
head_masked_scores = head_scores
head_weights = softmax(head_masked_scores)
```

</details>

마스크를 제거해도 √dₖ 스케일링과 잔차 연결은 유지한다.

### logits에서 보인 차이

<div class="bpe-comparison-table" role="region" aria-label="causal mask 제거 전후 logits" tabindex="0">

| 토큰 | 정상 logits | 마스크 제거 logits | 제거 후 확률 |
| --- | ---: | ---: | ---: |
| ` My` | -65.2963 | -73.3538 | 51.86% |
| ` Your` | -66.5362 | -75.0407 | 9.60% |
| ` It` | -66.9374 | -75.4249 | 6.54% |

</div>

1위가 `,`에서 ` My`로 바뀌었고, ` My`의 확률은 `51.86%`였다. 최대 절대차는 `19.8112`였다.

첫 블록·첫 헤드에서 `Hello`의 참고 비율은 원래 `[1, 0, 0, 0, 0]`이었다. 마스크를 제거하자 뒤의 네 토큰을 참고하는 비율의 합이 `40.13%`가 됐다. **앞쪽 토큰이 뒤쪽 토큰의 정보까지 가져온 것**이다.

마지막 ` Is`는 원래도 입력 전체를 볼 수 있어 첫 블록의 출력이 같았다. 하지만 앞쪽 토큰의 벡터가 바뀌었고, **두 번째 블록부터 그 벡터를 참고하면서 마지막 위치의 출력도 달라졌다.**

### causal mask의 역할과 탐구 질문

**질문: 입력 문장을 이미 전부 알고 있는데도 미래 토큰을 가려야 할까?**

`Hello` 위치의 출력은 다음 토큰인 `!`를 예측한다. 이때 `!`를 참고할 수 있다면 정답을 미리 보고 예측하는 셈이다. causal mask는 문장 전체를 한 번에 계산해도 ==yellow:각 위치가 자기 위치까지의 정보만 쓰게 한다.==

실제로 `Hello`만 넣었을 때와 전체 문장을 넣었을 때 첫 위치의 logits를 비교하면, 정상 모델의 최대 차이는 약 `1.49 × 10⁻¹³`이지만 마스크 제거 모델은 `23.2884`였다. 뒤에 붙인 토큰이 앞 위치의 예측을 바꾼 것이다.

## 3. 잔차 연결 제거

### 바뀌는 코드

`forward()`에서 Attention 뒤의 `x`와 FFN 뒤의 `h`를 더하지 않는다. 두 연산의 출력만 다음 단계에 전달한다.

<details>
<summary>잔차 연결 변경 전·후 코드 보기</summary>

**변경 전**

```python
h = x + attention
```

```python
x = h + ffn_output
```

**변경 후**

```python
# h = x + attention
h = attention
```

```python
# x = h + ffn_output
x = ffn_output
```

</details>

두 덧셈 사이의 LayerNorm과 FFN은 그대로 실행한다. 스케일링과 causal mask도 유지한다.

### logits에서 보인 차이

<div class="bpe-comparison-table" role="region" aria-label="잔차 연결 제거 전후 logits" tabindex="0">

| 토큰 | 정상 logits | 잔차 제거 logits | 제거 후 확률 |
| --- | ---: | ---: | ---: |
| `\n` | -65.6975 | -106.7136 | 26.30% |
| `/` | -68.4270 | -107.1885 | 16.36% |
| `-` | -67.3038 | -107.3026 | 14.59% |

</div>

1위가 `,`에서 줄바꿈으로 바뀌고, `/`, `-`가 뒤를 이었다. 최대 절대차는 `84.3594`로 이번 세 실험 중 가장 컸다.

점수만 낮아진 것이 아니라 **상위 후보와 확률 분포도 함께 달라졌다.**

### 잔차 연결의 역할과 탐구 질문

**질문: Attention과 FFN의 출력에 왜 입력을 다시 더할까?**

Attention 출력은 여러 위치의 V를 참고 비율로 섞고 변환한 값이다. **이 값만 다음 단계로 넘기면 현재 토큰의 벡터를 섞인 정보로 대체하게 된다.** 비슷한 위치를 비슷한 비율로 참고하는 토큰들은 출력도 비슷해질 수 있어, 원래 토큰을 구분하던 특징이 약해질 수 있다.

`x + attention`은 ==blue:원래 토큰의 벡터를 직접 전달하는 경로==를 두고, 그 위에 참고한 정보를 더한다. 입력 정보가 Attention 출력에도 반영되지만, 잔차 연결이 없으면 그 정보가 모두 Attention의 변환을 거쳐야 한다.

GPT-2의 가중치는 잔차 연결이 있는 상태로 학습됐다. 가중치를 유지한 채 이 경로를 빼자 다음 토큰 분포가 크게 달라졌다. 이 입력의 결과만으로 잔차 제거가 항상 가장 큰 영향을 준다고 볼 수는 없다.

---

여기까지는 가중치를 고정하고 구성 요소를 바꿨다. 이어서는 **손실을 계산하고 학습에 사용하는 원리**를 살펴본다. 각 항목은 역할, 수식, 짧은 설명, 숫자 예시 순서로 정리했다.

SFT·RLHF·DPO의 숫자는 ==orange:설명용 가상 예제==다. 로그는 자연로그이며 수치는 반올림해 표시했다. 실제 GPT-2를 미세조정한 결과는 아니다.

> **이후 예시의 공통 질문과 답변**
>
> - 질문: `대한민국의 수도는?`
> - A — 모범·선호 답변: `서울입니다.`
> - B — 비선호 답변: `부산입니다.`

## 4. 다음 토큰 예측 손실

다음 토큰 예측 손실은 **모델이 실제 정답 토큰에 얼마나 높은 확률을 줬는지 평가하는 값**이다.

<div class="llm-equation" role="region" aria-label="다음 토큰 예측의 평균 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="다음 토큰 예측의 평균 손실">
  <mrow><mrow><mi>L</mi></mrow><mo>=</mo><mo>−</mo><mfrac><mn>1</mn><mrow><mi>N</mi></mrow></mfrac><mrow><munderover><mo>∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>N</mi></mrow></munderover><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mrow><msub><mi>p</mi><mi>t</mi></msub></mrow><mo>)</mo></mrow></mrow></mrow>
</math>
</div>

1. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="pₜ"><mrow class="text-highlight" data-math-variable="p_t" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>t</mi></msub></mrow></math>는 각 위치의 **정답 토큰 확률**이다. 여기에 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="−ln"><mrow><mo>−</mo><mi mathvariant="normal">ln</mi></mrow></math>을 취하면 ==yellow:정답 확률이 낮을수록 손실이 커진다.==
2. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="Σ"><mo>∑</mo></math>로 위치별 손실을 더하고, 토큰 수 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="N"><mrow class="text-highlight" data-math-variable="N" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>N</mi></mrow></math>으로 나눠 평균을 구한다. 정답이 one-hot일 때의 교차 엔트로피와 같은 계산이다.

<div class="llm-example" role="note" aria-label="다음 토큰 예측 손실 계산 예시">

**예시** · 두 토큰의 평균 손실

답변을 `[서울] [입니다]` 두 토큰으로 가정하고, 각 정답 확률이 0.5와 0.25라고 하자. 실제 GPT-2 토큰화 결과는 아니다.

<div class="llm-equation" role="region" aria-label="첫 번째 정답 토큰의 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="첫 번째 정답 토큰의 손실">
<mrow><mrow><msub><mi>L</mi><mn>1</mn></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.5</mn><mo>)</mo></mrow><mo>≈</mo><mrow><mn>0.693147</mn></mrow></mrow>
</math>
</div>

`서울`에 준 확률 0.5를 첫 번째 위치의 손실로 바꾼 값이다.

<div class="llm-equation" role="region" aria-label="두 번째 정답 토큰의 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="두 번째 정답 토큰의 손실">
<mrow><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.25</mn><mo>)</mo></mrow><mo>≈</mo><mrow><mn>1.386294</mn></mrow></mrow>
</math>
</div>

`입니다`에는 더 낮은 확률을 줬으므로 손실이 더 크다.

<div class="llm-equation" role="region" aria-label="두 위치 손실의 평균" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="두 위치 손실의 평균">
<mrow><mrow><mi>L</mi></mrow><mo>=</mo><mfrac><mrow><mn>0.693147</mn><mo>+</mo><mn>1.386294</mn></mrow><mn>2</mn></mfrac><mo>≈</mo><mrow><mn>1.039721</mn></mrow></mrow>
</math>
</div>

두 손실을 더하고 토큰 수 2로 나눈 **토큰당 평균 손실**이다.

</div>

같은 계산을 기존 GPT-2의 `Hello! My Name Is` 출력에도 적용했다. 네 개의 실제 다음 토큰 `!`, ` My`, ` Name`, ` Is`를 정답으로 평가한 평균 손실은 3.587852였다.

## 5. SFT: 모범 답변을 따라 배우기

SFT는 **질문과 모범 답변을 주고, 그 답변이 나오도록 모델을 추가 학습하는 것**이다.

<div class="llm-equation" role="region" aria-label="응답 마스크를 적용한 SFT 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="응답 마스크를 적용한 SFT 손실">
  <mrow><mrow><msub><mi>L</mi><mtext>SFT</mtext></msub></mrow><mo>=</mo><mo>−</mo><mfrac><mrow><msub><mo>∑</mo><mi>t</mi></msub><mrow><mrow><msub><mi>m</mi><mi>t</mi></msub></mrow><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mrow><msub><mi>p</mi><mi>t</mi></msub></mrow><mo>)</mo></mrow></mrow></mrow><mrow><msub><mo>∑</mo><mi>t</mi></msub><mrow><msub><mi>m</mi><mi>t</mi></msub></mrow></mrow></mfrac></mrow>
</math>
</div>

1. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="pₜ"><mrow class="text-highlight" data-math-variable="p_t" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>t</mi></msub></mrow></math>는 정답 토큰 확률, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="mₜ"><mrow class="text-highlight" data-math-variable="m_t" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>m</mi><mi>t</mi></msub></mrow></math>는 손실에 포함할 위치의 마스크다. 여기서는 ==blue:답변에 1, 프롬프트에 0==을 둔다.
2. 마스크를 곱한 손실을 더하고, **포함한 답변 토큰 수**로 나눈다. 앞선 문맥에는 실제 모범 답변 토큰을 넣는다.

<div class="llm-example" role="note" aria-label="SFT 손실 계산 예시">

**예시** · 마스크로 답변 손실만 남기기

질문과 답변을 `[대한민국의] [수도는?] [서울] [입니다]`로 단순화해 보자. 마스크는 **입력 토큰이 아니라 예측할 정답**을 기준으로 붙인다.

| 예측할 정답 | 정답 확률 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="pₜ"><mrow><msub><mi>p</mi><mi>t</mi></msub></mrow></math> | 마스크 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="mₜ"><mrow><msub><mi>m</mi><mi>t</mi></msub></mrow></math> |
| --- | ---: | ---: |
| 수도는? — 질문 내부 | 0.3 | 0 |
| 서울 — 답변 | 0.5 | 1 |
| 입니다 — 답변 | 0.25 | 1 |

<div class="llm-equation" role="region" aria-label="질문 위치의 손실을 마스크로 제외" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="질문 위치의 손실을 마스크로 제외">
<mrow><mn>0</mn><mo>×</mo><mrow><mo>(</mo><mrow><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.3</mn><mo>)</mo></mrow></mrow><mo>)</mo></mrow><mo>=</mo><mrow><mn>0</mn></mrow></mrow>
</math>
</div>

질문 내부 토큰은 마스크가 0이므로, 그 위치의 손실이 합계에서 빠진다.

<div class="llm-equation" role="region" aria-label="마스크가 1인 답변 손실의 합" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="마스크가 1인 답변 손실의 합">
<mrow><mtext>손실 합</mtext><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.5</mn><mo>)</mo></mrow><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.25</mn><mo>)</mo></mrow><mo>≈</mo><mrow><mn>2.079442</mn></mrow></mrow>
</math>
</div>

답변 두 위치는 마스크가 1이어서 손실이 그대로 남는다.

<div class="llm-equation" role="region" aria-label="포함한 답변 토큰 수로 평균" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="포함한 답변 토큰 수로 평균">
<mrow><mrow><msub><mi>L</mi><mtext>SFT</mtext></msub></mrow><mo>≈</mo><mfrac><mn>2.079442</mn><mrow><mn>0</mn><mo>+</mo><mn>1</mn><mo>+</mo><mn>1</mn></mrow></mfrac><mo>=</mo><mrow><mn>1.039721</mn></mrow></mrow>
</math>
</div>

분모는 전체 예측 위치 3개가 아니라 **포함한 답변 토큰 2개**다. 프롬프트 마지막 위치에서 첫 답변을 예측하는 손실도 포함한다.

</div>

## 6. RLHF: 사람이 선호한 답변을 더 자주 만들기

RLHF는 **사람의 피드백을 보상으로 연결해 언어 모델을 학습하는 것**이다. 여기서는 보상 모델을 학습한 뒤 PPO를 사용하는 대표적인 흐름을 다룬다.

계산은 [보상 모델](#61-보상-모델-손실) → [기대 보상](#62-기대-보상)·[KL](#63-kl-페널티) → [Advantage](#64-advantage) → [PPO](#65-ppo-정책-손실)·[가치 모델](#66-가치-모델-손실) 순서로 이어진다.

### 6.1. 보상 모델 손실

보상 모델 손실은 **사람이 고른 답변에 상대적으로 높은 점수를 주도록 학습하는 기준**이다.

<div class="llm-equation" role="region" aria-label="보상 모델의 선호 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="보상 모델의 선호 손실">
  <mrow><mrow><msub><mi>L</mi><mtext>RM</mtext></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mi mathvariant="normal">σ</mi><mrow><mo>(</mo><mrow><mrow><msub><mi>r</mi><mi>A</mi></msub></mrow><mo>−</mo><mrow><msub><mi>r</mi><mi>B</mi></msub></mrow></mrow><mo>)</mo></mrow></mrow>
</math>
</div>

<div class="llm-equation" role="region" aria-label="Sigmoid 함수" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Sigmoid 함수">
  <mrow><mi mathvariant="normal">σ</mi><mrow><mo>(</mo><mrow><mi>z</mi></mrow><mo>)</mo></mrow><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mrow><mi>z</mi></mrow></mrow></msup></mrow></mfrac></mrow>
</math>
</div>

1. 선호 답변의 점수 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="r 아래첨자 A"><mrow class="text-highlight" data-math-variable="r_A" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>r</mi><mi>A</mi></msub></mrow></math>에서 비선호 답변의 점수 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="r 아래첨자 B"><mrow class="text-highlight" data-math-variable="r_B" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>r</mi><mi>B</mi></msub></mrow></math>를 뺀다.
2. sigmoid <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="σ"><mi>σ</mi></math>는 점수 차이 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="z"><mrow class="text-highlight" data-math-variable="z" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>z</mi></mrow></math>를 0~1의 선호 확률로 바꾼다. 식의 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="e"><mi>e</mi></math>는 약 2.718이다.
3. 선호 확률에 음의 로그를 취한다. 이 손실로 ==blue:보상 모델을 학습한다.==

<div class="llm-example" role="note" aria-label="보상 모델의 선호 손실 계산 예시">

**예시** · 보상 점수에서 선호 확률로

사람은 A를 골랐고, 보상 모델의 점수는 A가 2, B가 1이라고 하자.

<div class="llm-equation" role="region" aria-label="선호 답변과 비선호 답변의 보상 차이" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="선호 답변과 비선호 답변의 보상 차이">
<mrow><mrow><msub><mi>r</mi><mi>A</mi></msub></mrow><mo>−</mo><mrow><msub><mi>r</mi><mi>B</mi></msub></mrow><mo>=</mo><mn>2</mn><mo>−</mo><mn>1</mn><mo>=</mo><mrow><mn>1</mn></mrow></mrow>
</math>
</div>

보상 모델이 선호 답변 A에 B보다 1점 높은 점수를 준 상태다.

<div class="llm-equation" role="region" aria-label="보상 차이를 사람의 선호 확률로 변환" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="보상 차이를 사람의 선호 확률로 변환">
<mrow><mi mathvariant="normal">σ</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></mfrac><mo>≈</mo><mrow><mn>0.731059</mn></mrow></mrow>
</math>
</div>

점수 차이를 사람이 A를 선호할 **예측 확률**로 바꾼다.

<div class="llm-equation" role="region" aria-label="실제 선택의 확률에서 보상 모델 손실 계산" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="실제 선택의 확률에서 보상 모델 손실 계산">
<mrow><mrow><msub><mi>L</mi><mtext>RM</mtext></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mi mathvariant="normal">σ</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow><mo>≈</mo><mrow><mn>0.313262</mn></mrow></mrow>
</math>
</div>

실제 선택이 A였으므로 그 선택의 확률에 음의 로그를 취한다. A의 선호를 더 잘 예측할수록 이 손실은 작아진다.

</div>

### 6.2. 기대 보상

기대 보상은 **현재 모델이 답변을 생성할 때 받을 보상의 평균적인 값**이다.

<div class="llm-equation" role="region" aria-label="정책의 기대 보상" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="정책의 기대 보상">
  <mrow><mrow><mi mathvariant="normal">E</mi><mo>[</mo><mrow><mi>r</mi></mrow><mo>]</mo></mrow><mo>=</mo><mrow><msub><mo>∑</mo><mrow><mi>y</mi></mrow></msub><mrow><mrow><msub><mi>p</mi><mi>y</mi></msub></mrow><mrow><msub><mi>r</mi><mi>y</mi></msub></mrow></mrow></mrow></mrow>
</math>
</div>

1. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 y"><mrow class="text-highlight" data-math-variable="p_y" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>y</mi></msub></mrow></math>는 답변 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="답변 y"><mrow class="text-highlight" data-math-variable="y" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>y</mi></mrow></math>를 생성할 확률, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="r 아래첨자 y"><mrow class="text-highlight" data-math-variable="r_y" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>r</mi><mi>y</mi></msub></mrow></math>는 그 답변의 보상이다.
2. ==yellow:확률과 보상을 곱해 모두 더한다.== <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="E[r]"><mrow><mi mathvariant="normal">E</mi><mo>[</mo><mrow><mi>r</mi></mrow><mo>]</mo></mrow></math>는 이렇게 구한 기댓값을 뜻한다.

<div class="llm-example" role="note" aria-label="기대 보상 계산 예시">

**예시** · 두 답변의 평균 보상

가능한 답변이 A와 B뿐이고, 확률은 0.6과 0.4, 보상은 2와 1이라고 하자.

<div class="llm-equation" role="region" aria-label="A가 기대 보상에 기여하는 값" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="A가 기대 보상에 기여하는 값">
<mrow><mn>0.6</mn><mo>×</mo><mn>2</mn><mo>=</mo><mrow><mn>1.2</mn></mrow></mrow>
</math>
</div>

A가 나올 확률과, A가 나왔을 때 받을 점수를 함께 반영한 값이다.

<div class="llm-equation" role="region" aria-label="B가 기대 보상에 기여하는 값" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="B가 기대 보상에 기여하는 값">
<mrow><mn>0.4</mn><mo>×</mo><mn>1</mn><mo>=</mo><mrow><mn>0.4</mn></mrow></mrow>
</math>
</div>

같은 방식으로 B가 평균 보상에 기여하는 부분을 계산한다.

<div class="llm-equation" role="region" aria-label="두 답변의 기여를 합친 기대 보상" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="두 답변의 기여를 합친 기대 보상">
<mrow><mrow><mi mathvariant="normal">E</mi><mo>[</mo><mrow><mi>r</mi></mrow><mo>]</mo></mrow><mo>=</mo><mn>1.2</mn><mo>+</mo><mn>0.4</mn><mo>=</mo><mrow><mn>1.6</mn></mrow></mrow>
</math>
</div>

두 기여를 더하면 현재 모델이 답변을 생성할 때 받을 **평균적인 보상**이 된다.

</div>

### 6.3. KL 페널티

KL 페널티는 보상을 높이려다 ==yellow:기준 모델과 크게 달라지는 변화에 비용을 붙이는 것==이다.

<div class="llm-equation" role="region" aria-label="현재 모델과 기준 모델 사이의 KL" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="현재 모델과 기준 모델 사이의 KL">
  <mrow><mrow><msub><mi>D</mi><mtext>KL</mtext></msub></mrow><mo>=</mo><mrow><msub><mo>∑</mo><mrow><mi>y</mi></mrow></msub><mrow><mrow><msub><mi>p</mi><mi>y</mi></msub></mrow><mi mathvariant="normal">ln</mi><mfrac><mrow><msub><mi>p</mi><mi>y</mi></msub></mrow><mrow><msub><mi>q</mi><mi>y</mi></msub></mrow></mfrac></mrow></mrow></mrow>
</math>
</div>

<div class="llm-equation" role="region" aria-label="보상에서 KL 페널티를 뺀 최대화 목표" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="보상에서 KL 페널티를 뺀 최대화 목표">
  <mrow><mrow><mi>J</mi></mrow><mo>=</mo><mrow><mi mathvariant="normal">E</mi><mo>[</mo><mrow><mi>r</mi></mrow><mo>]</mo></mrow><mo>−</mo><mrow><mi>λ</mi></mrow><mrow><msub><mi>D</mi><mtext>KL</mtext></msub></mrow></mrow>
</math>
</div>

1. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 y"><mrow class="text-highlight" data-math-variable="p_y" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>y</mi></msub></mrow></math>는 현재 모델, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="q 아래첨자 y"><mrow class="text-highlight" data-math-variable="q_y" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>q</mi><mi>y</mi></msub></mrow></math>는 고정된 기준 모델의 답변 확률이다. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 y / q 아래첨자 y"><mfrac><mrow><msub><mi>p</mi><mi>y</mi></msub></mrow><mrow><msub><mi>q</mi><mi>y</mi></msub></mrow></mfrac></math>는 기준에 비해 그 답변의 확률이 몇 배가 됐는지 나타낸다.
2. 로그 비율은 **두 음의 로그 손실의 차이**다.

   <div class="llm-equation" role="region" aria-label="로그 비율을 두 음의 로그 손실의 차이로 읽기" tabindex="0">
   <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="로그 확률 비율은 두 음의 로그 확률의 차이"><mrow><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mfrac><mrow><msub><mi>p</mi><mi>y</mi></msub></mrow><mrow><msub><mi>q</mi><mi>y</mi></msub></mrow></mfrac><mo>)</mo></mrow><mo>=</mo><mrow><mo>(</mo><mrow><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mrow><msub><mi>q</mi><mi>y</mi></msub></mrow><mo>)</mo></mrow></mrow><mo>)</mo></mrow><mo>−</mo><mrow><mo>(</mo><mrow><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mrow><msub><mi>p</mi><mi>y</mi></msub></mrow><mo>)</mo></mrow></mrow><mo>)</mo></mrow></mrow></math>
   </div>

   현재 모델이 내는 답변을 기준 모델의 확률로 평가할 때, **음의 로그 손실이 얼마나 더 드는지** 비교한다.
3. 현재 모델이 자주 내는 답변을 더 반영하도록 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 y"><mrow class="text-highlight" data-math-variable="p_y" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>y</mi></msub></mrow></math>를 곱해 평균낸다. 그 비용을 기대 보상에서 뺀 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="J"><mrow class="text-highlight" data-math-variable="J" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>J</mi></mrow></math>를 최대화한다. 

**페널티가 없다면** 보상 점수만 좇으며 기준 모델의 행동에서 크게 벗어날 수 있다. 예를 들어 보상 모델이 긴 답변을 좋아하면, 모델이 불필요한 반복으로 점수를 올릴 수 있다.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="λ"><mrow class="text-highlight" data-math-variable="λ" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>λ</mi></mrow></math>는 ==blue:KL 비용에 얼마나 큰 비중을 둘지 정하는 계수==다. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="λ = 0"><mrow><mrow><mi>λ</mi></mrow><mo>=</mo><mn>0</mn></mrow></math>이면 페널티가 없고, 클수록 기준 모델을 강하게 유지하려 하며, 작을수록 보상을 높이기 위한 변화를 더 많이 허용한다.

<div class="llm-example" role="note" aria-label="KL 페널티 계산 예시">

**예시** · 분포 변화에 비용 붙이기

현재 확률은 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="현재 확률 0.6과 0.4"><mrow><mo>[</mo><mn>0.6</mn><mo>,</mo><mn>0.4</mn><mo>]</mo></mrow></math>, 기준 확률은 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="기준 확률 0.5와 0.5"><mrow><mo>[</mo><mn>0.5</mn><mo>,</mo><mn>0.5</mn><mo>]</mo></mrow></math>라고 하자.

<div class="llm-equation" role="region" aria-label="A의 확률 변화가 KL에 기여하는 항" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="A의 확률 변화가 KL에 기여하는 항">
<mrow><mn>0.6</mn><mo>×</mo><mi mathvariant="normal">ln</mi><mfrac><mn>0.6</mn><mn>0.5</mn></mfrac><mo>≈</mo><mrow><mn>0.109393</mn></mrow></mrow>
</math>
</div>

A의 확률은 기준의 1.2배다. 그 로그 변화에 현재 A의 생성 확률을 곱해 반영한다.

<div class="llm-equation" role="region" aria-label="B의 확률 변화가 KL에 기여하는 항" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="B의 확률 변화가 KL에 기여하는 항">
<mrow><mn>0.4</mn><mo>×</mo><mi mathvariant="normal">ln</mi><mfrac><mn>0.4</mn><mn>0.5</mn></mfrac><mo>≈</mo><mrow><mo>−</mo><mn>0.089257</mn></mrow></mrow>
</math>
</div>

B의 확률은 기준의 0.8배라서 로그가 음수다. **개별 항은 음수가 될 수 있다.**

<div class="llm-equation" role="region" aria-label="두 항을 합산한 KL" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="두 항을 합산한 KL">
<mrow><mrow><msub><mi>D</mi><mtext>KL</mtext></msub></mrow><mo>≈</mo><mn>0.109393</mn><mo>+</mo><mrow><mo>(</mo><mrow><mo>−</mo><mn>0.089257</mn></mrow><mo>)</mo></mrow><mo>≈</mo><mrow><mn>0.020136</mn></mrow></mrow>
</math>
</div>

두 항을 더한 값이 현재 분포와 기준 분포의 차이다. 전체 KL은 0 이상이다.

앞의 기대 보상 1.6과 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="람다 1"><mrow><mrow><mi>λ</mi></mrow><mo>=</mo><mn>1</mn></mrow></math>을 사용하면:

<div class="llm-equation" role="region" aria-label="기대 보상에서 KL 비용을 뺀 목표값" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="기대 보상에서 KL 비용을 뺀 목표값">
<mrow><mrow><mi>J</mi></mrow><mo>=</mo><mn>1.6</mn><mo>−</mo><mn>1</mn><mo>×</mo><mn>0.020136</mn><mo>≈</mo><mrow><mn>1.579864</mn></mrow></mrow>
</math>
</div>

기대 보상에서 분포 변화의 비용을 뺀 **최대화할 목표값**이다. 여기서는 페널티 강도가 1이므로 KL 값만큼 뺀다.

</div>

KL은 두 분포가 같으면 0이다. 답변의 정확성을 직접 판정하는 값은 아니다.

### 6.4. Advantage

Advantage는 **어떤 선택의 결과가 예상보다 얼마나 좋았는지 나타내는 값**이다. 한 번 선택하고 끝나는 예제에서는 다음처럼 구한다.

<div class="llm-equation" role="region" aria-label="한 번의 선택에서 advantage 추정" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="한 번의 선택에서 advantage 추정">
  <mrow><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo>=</mo><mrow><mi>G</mi></mrow><mo>−</mo><mrow><mi>V</mi></mrow></mrow>
</math>
</div>

1. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="G"><mrow class="text-highlight" data-math-variable="G" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>G</mi></mrow></math>는 이후의 보상과 보정을 반영한 return, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="V"><mrow class="text-highlight" data-math-variable="V" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>V</mi></mrow></math>는 가치 모델의 예상값이다. 실제 결과에서 예상값을 뺀다.
2. ==yellow:양수면 예상보다 좋았고, 음수면 못했다==는 뜻이다. 실제 여러 토큰의 PPO에서는 보통 GAE로 추정한다.

<div class="llm-example" role="note" aria-label="Advantage 계산 예시">

**예시** · 실제 결과와 예상의 차이

별도의 한 번의 선택 예제로, 보정된 return이 2이고 예상값이 1.5라고 하자.

<div class="llm-equation" role="region" aria-label="Return과 예상값의 차이로 advantage 계산" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Return과 예상값의 차이로 advantage 계산">
<mrow><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo>=</mo><mn>2</mn><mo>−</mo><mn>1.5</mn><mo>=</mo><mrow><mn>0.5</mn></mrow></mrow>
</math>
</div>

예상보다 0.5만큼 좋은 결과였으므로, **같은 선택의 확률을 높이고 싶은 상황**이라는 신호다.

</div>

### 6.5. PPO 정책 손실

PPO 정책 손실은 **좋은 선택의 확률을 높이되, 한 번 모은 데이터로 지나치게 바뀌는 것을 조절하는 기준**이다.

<div class="llm-equation" role="region" aria-label="현재 정책과 이전 정책의 확률 비율" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="현재 정책과 이전 정책의 확률 비율">
  <mrow><mrow><mi>ρ</mi></mrow><mo>=</mo><mfrac><mrow><msub><mi>p</mi><mtext>current</mtext></msub></mrow><mrow><msub><mi>p</mi><mtext>old</mtext></msub></mrow></mfrac></mrow>
</math>
</div>

<div class="llm-equation" role="region" aria-label="PPO의 한 선택에 대한 정책 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="PPO의 한 선택에 대한 정책 손실">
  <mrow><mrow><msub><mi>L</mi><mtext>actor</mtext></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">min</mi><mrow><mo>(</mo><mrow><mrow><mi>ρ</mi></mrow><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo>,</mo><mi mathvariant="normal">clip</mi><mrow><mo>(</mo><mrow><mrow><mi>ρ</mi></mrow><mo>,</mo><mn>1</mn><mo>−</mo><mrow><mi>ε</mi></mrow><mo>,</mo><mn>1</mn><mo>+</mo><mrow><mi>ε</mi></mrow></mrow><mo>)</mo></mrow><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow></mrow><mo>)</mo></mrow></mrow>
</math>
</div>

1. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="ρ"><mrow class="text-highlight" data-math-variable="ρ" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>ρ</mi></mrow></math>는 같은 선택의 현재 확률을 이전 정책의 확률로 나눈 비율이고, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="Â"><mrow class="text-highlight" data-math-variable="A-hat" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow></math>는 advantage다.
2. <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="clip"><mi mathvariant="normal">clip</mi></math>은 비율을 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="[1 − ε, 1 + ε]"><mrow><mo>[</mo><mn>1</mn><mo>−</mo><mrow><mi>ε</mi></mrow><mo>,</mo><mn>1</mn><mo>+</mo><mrow><mi>ε</mi></mrow><mo>]</mo></mrow></math>로 자른다. 원래 비율과 자른 비율에 각각 advantage를 곱한다.
3. 두 값 중 작은 값을 고르고 마이너스를 붙인다. 좋은 방향으로 충분히 변한 선택에는 ==yellow:추가 이득을 주지 않는== 방식이다.

<div class="llm-example" role="note" aria-label="PPO 정책 손실 계산 예시">

**예시** · 확률 비율과 clipping

이전 확률은 0.6, 현재 확률은 0.78이고, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="advantage 0.5"><mrow><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo>=</mo><mn>0.5</mn></mrow></math>, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="epsilon 0.2"><mrow><mrow><mi>ε</mi></mrow><mo>=</mo><mn>0.2</mn></mrow></math>라고 하자.

<div class="llm-equation" role="region" aria-label="이전 정책 대비 현재 확률 비율" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="이전 정책 대비 현재 확률 비율">
<mrow><mrow><mi>ρ</mi></mrow><mo>=</mo><mfrac><mn>0.78</mn><mn>0.6</mn></mfrac><mo>=</mo><mrow><mn>1.3</mn></mrow></mrow>
</math>
</div>

이 선택의 생성 확률이 이전 정책의 1.3배가 됐다는 뜻이다.

<div class="llm-equation" role="region" aria-label="Clipping으로 계산용 확률 비율 제한" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Clipping으로 계산용 확률 비율 제한">
<mrow><mi mathvariant="normal">clip</mi><mrow><mo>(</mo><mrow><mn>1.3</mn><mo>,</mo><mn>0.8</mn><mo>,</mo><mn>1.2</mn></mrow><mo>)</mo></mrow><mo>=</mo><mrow><mn>1.2</mn></mrow></mrow>
</math>
</div>

설정한 상한 1.2를 넘었으므로, 비교에 사용할 두 번째 비율은 1.2로 자른다.

<div class="llm-equation" role="region" aria-label="원래 비율과 자른 비율에 advantage 반영" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="원래 비율과 자른 비율에 advantage 반영">
<mtable columnalign="left"><mtr><mtd><mrow><mn>1.3</mn><mo>×</mo><mn>0.5</mn><mo>=</mo><mrow><mn>0.65</mn></mrow></mrow></mtd></mtr><mtr><mtd><mrow><mn>1.2</mn><mo>×</mo><mn>0.5</mn><mo>=</mo><mrow><mn>0.60</mn></mrow></mrow></mtd></mtr></mtable>
</math>
</div>

좋은 결과였다는 신호를 두 비율에 곱해, 원래 목표와 clipping한 목표를 비교한다.

<div class="llm-equation" role="region" aria-label="작은 목표값을 선택한 PPO 정책 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="작은 목표값을 선택한 PPO 정책 손실">
<mrow><mrow><msub><mi>L</mi><mtext>actor</mtext></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">min</mi><mrow><mo>(</mo><mrow><mn>0.65</mn><mo>,</mo><mn>0.60</mn></mrow><mo>)</mo></mrow><mo>=</mo><mrow><mo>−</mo><mn>0.60</mn></mrow></mrow>
</math>
</div>

작은 값 0.60을 고르고, 최소화할 손실로 바꾸기 위해 부호를 뒤집는다. 이 항에서는 비율을 더 높여도 추가 이득이 없다.

</div>

이전 정책은 데이터를 생성한 모델이고, KL의 기준 모델은 고정된 SFT 모델이다. clipping은 실제 확률을 강제로 구간 안에 고정하지 않는다.

### 6.6. 가치 모델 손실

가치 모델 손실은 **앞으로 받을 return을 더 정확히 예상하도록 학습하는 기준**이다.

<div class="llm-equation" role="region" aria-label="가치 모델의 한 샘플 제곱 오차 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="가치 모델의 한 샘플 제곱 오차 손실">
  <mrow><mrow><msub><mi>L</mi><mtext>value</mtext></msub></mrow><mo>=</mo><msup><mrow><mo>(</mo><mrow><mrow><mi>V</mi></mrow><mo>−</mo><mrow><mi>G</mi></mrow></mrow><mo>)</mo></mrow><mn>2</mn></msup></mrow>
</math>
</div>

1. 예측값 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="V"><mrow class="text-highlight" data-math-variable="V" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>V</mi></mrow></math>에서 목표 return <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="G"><mrow class="text-highlight" data-math-variable="G" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>G</mi></mrow></math>를 뺀 뒤 제곱한다.
2. ==yellow:예측이 목표에 가까워질수록 손실이 작아진다.== 여러 경험에서는 이 값을 평균한다.

<div class="llm-example" role="note" aria-label="가치 모델 손실 계산 예시">

**예시** · 예측 오차를 손실로

예측이 1.5이고 목표가 2라면:

<div class="llm-equation" role="region" aria-label="가치 예측과 실제 목표 사이의 오차" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="가치 예측과 실제 목표 사이의 오차">
<mrow><mrow><mi>V</mi></mrow><mo>−</mo><mrow><mi>G</mi></mrow><mo>=</mo><mn>1.5</mn><mo>−</mo><mn>2</mn><mo>=</mo><mrow><mo>−</mo><mn>0.5</mn></mrow></mrow>
</math>
</div>

이번 결과를 실제 목표보다 0.5 낮게 예상했다는 뜻이다.

<div class="llm-equation" role="region" aria-label="오차를 제곱한 가치 모델 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="오차를 제곱한 가치 모델 손실">
<mrow><mrow><msub><mi>L</mi><mtext>value</mtext></msub></mrow><mo>=</mo><msup><mrow><mo>(</mo><mrow><mo>−</mo><mn>0.5</mn></mrow><mo>)</mo></mrow><mn>2</mn></msup><mo>=</mo><mrow><mn>0.25</mn></mrow></mrow>
</math>
</div>

오차를 제곱해 크기를 평가한다. 이 손실을 줄여 가치 모델이 이번 목표에 더 가까운 값을 예상하도록 학습한다.

</div>

KL을 보상에 반영했다면 return과 advantage에도 그 영향이 들어간다. 같은 페널티를 무조건 다시 더하지 않도록 구분한다.

## 7. DPO: 두 답변의 선호를 직접 배우기

DPO는 **사람이 선호한 답변 쪽으로 상대적인 확률 비율이 이동하도록 언어 모델을 직접 학습하는 것**이다.

<div class="llm-equation" role="region" aria-label="기준 모델 대비 선호와 비선호의 로그 비율 차이" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="기준 모델 대비 선호와 비선호의 로그 비율 차이">
  <mrow><mrow><mi>d</mi></mrow><mo>=</mo><mi mathvariant="normal">ln</mi><mfrac><mrow><msub><mi>p</mi><mi>A</mi></msub></mrow><mrow><msub><mi>q</mi><mi>A</mi></msub></mrow></mfrac><mo>−</mo><mi mathvariant="normal">ln</mi><mfrac><mrow><msub><mi>p</mi><mi>B</mi></msub></mrow><mrow><msub><mi>q</mi><mi>B</mi></msub></mrow></mfrac></mrow>
</math>
</div>

<div class="llm-equation" role="region" aria-label="DPO 선호 손실" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="DPO 선호 손실">
  <mrow><mrow><msub><mi>L</mi><mtext>DPO</mtext></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mi mathvariant="normal">σ</mi><mrow><mo>(</mo><mrow><mrow><mi>β</mi></mrow><mrow><mi>d</mi></mrow></mrow><mo>)</mo></mrow></mrow>
</math>
</div>

여기서 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 A"><mrow class="text-highlight" data-math-variable="p_A" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>A</mi></msub></mrow></math>, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 B"><mrow class="text-highlight" data-math-variable="p_B" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>p</mi><mi>B</mi></msub></mrow></math>는 현재 모델, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="q 아래첨자 A"><mrow class="text-highlight" data-math-variable="q_A" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>q</mi><mi>A</mi></msub></mrow></math>, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="q 아래첨자 B"><mrow class="text-highlight" data-math-variable="q_B" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><msub><mi>q</mi><mi>B</mi></msub></mrow></math>는 기준 모델의 선호·비선호 답변 확률이다.

1. ==blue:기준 대비 변화:== <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 A / q 아래첨자 A"><mfrac><mrow><msub><mi>p</mi><mi>A</mi></msub></mrow><mrow><msub><mi>q</mi><mi>A</mi></msub></mrow></mfrac></math>와 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="p 아래첨자 B / q 아래첨자 B"><mfrac><mrow><msub><mi>p</mi><mi>B</mi></msub></mrow><mrow><msub><mi>q</mi><mi>B</mi></msub></mrow></mfrac></math>로 각 답변의 확률이 기준의 몇 배가 됐는지 본다.
2. ==blue:두 변화의 비교:== 두 비율의 로그 차이 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="d"><mrow class="text-highlight" data-math-variable="d" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>d</mi></mrow></math>는 선호 답변 쪽으로 상대적인 비율이 얼마나 이동했는지 나타낸다. 두 변화율이 같으면 0, 선호 쪽 변화율이 더 크면 양수다.
3. ==blue:선호 확률로 변환:== 양수 계수 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="β"><mrow class="text-highlight" data-math-variable="β" style="--highlight-light: #d5e8fa; --highlight-dark: #2e4b65; padding: 0;"><mi>β</mi></mrow></math>로 차이의 척도를 조절하고 sigmoid를 적용한다. 이 값은 현재·기준 모델의 확률 관계가 **A를 선호하는 비교 결과를 얼마나 지지하는지** 나타낸다.
4. ==blue:손실로 변환:== 그 확률에 음의 로그를 취한다. A의 선호를 더 잘 지지할수록 손실이 작아진다. 이 손실로 ==yellow:현재 모델만 학습한다.==

<div class="llm-example" role="note" aria-label="DPO 손실 계산 예시">

**예시** · 두 답변의 변화 비율 비교

A의 현재·기준 확률은 0.6·0.3, B는 0.2·0.4이고, <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="β = 1"><mrow><mrow><mi>β</mi></mrow><mo>=</mo><mn>1</mn></mrow></math>로 둔다.

A의 변화율은 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="선호 답변의 확률은 기준의 두 배"><mrow><mfrac><mn>0.6</mn><mn>0.3</mn></mfrac><mo>=</mo><mrow><mn>2</mn></mrow></mrow></math>, B는 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="비선호 답변의 확률은 기준의 절반"><mrow><mfrac><mn>0.2</mn><mn>0.4</mn></mfrac><mo>=</mo><mrow><mn>0.5</mn></mrow></mrow></math>다. **기준에 비해 A는 2배, B는 절반**이 됐다.

<div class="llm-equation" role="region" aria-label="두 변화율의 로그 차이" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="두 변화율의 로그 차이">
<mtable columnalign="left"><mtr><mtd><mrow><mrow><mi>d</mi></mrow><mo>=</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.5</mn><mo>)</mo></mrow><mo>=</mo><mrow><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>4</mn><mo>)</mo></mrow></mrow></mrow></mtd></mtr></mtable>
</math>
</div>

두 변화율의 비율은 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" aria-label="2 나누기 0.5는 4"><mrow><mfrac><mn>2</mn><mn>0.5</mn></mfrac><mo>=</mo><mrow><mn>4</mn></mrow></mrow></math>다. 즉 **A와 B의 생성 확률 비율이 기준 모델의 4배**가 됐다는 뜻이다.

<div class="llm-equation" role="region" aria-label="확률 관계가 선호를 지지하는 정도" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="확률 관계가 선호를 지지하는 정도">
<mtable columnalign="left"><mtr><mtd><mrow><mi mathvariant="normal">σ</mi><mrow><mo>(</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>4</mn><mo>)</mo></mrow><mo>)</mo></mrow><mo>=</mo><mrow><mn>0.8</mn></mrow></mrow></mtd></mtr></mtable>
</math>
</div>

이 차이를 sigmoid에 넣으면 선호 관계에 부여한 확률이 0.8이 된다. A 자체의 생성 확률 0.6과는 다르다.

<div class="llm-equation" role="region" aria-label="선호 확률을 손실로 변환" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="선호 확률을 손실로 변환">
<mtable columnalign="left"><mtr><mtd><mrow><mrow><msub><mi>L</mi><mtext>DPO</mtext></msub></mrow><mo>=</mo><mo>−</mo><mi mathvariant="normal">ln</mi><mrow><mo>(</mo><mn>0.8</mn><mo>)</mo></mrow><mo>≈</mo><mrow><mn>0.223144</mn></mrow></mrow></mtd></mtr></mtable>
</math>
</div>

선호 관계에 높은 확률을 줬으므로 손실은 약 0.223144다. 이 값을 더 낮추는 방향으로 모델을 학습한다.


</div>

여러 토큰의 답변은 응답 토큰의 **로그 확률을 합산**해 평가한다. 원래 DPO에서는 이 합을 토큰 평균으로 바꾸지 않는다. 기준 모델은 고정하며, 직접 학습하는 대상은 두 답변의 상대적인 관계다.

## 마치며

최근 학교에서 인공지능 수학을 공부하고 있는데, 이번 활동은 그 수학을 왜 배우는지 한 번에 와닿게 해주는 활동이었다. 행렬 연산으로 모델의 출력을 만들고, 확률을 이용해 손실을 계산하는 과정을 따라가면서 ==yellow:수업에서 배우는 내용이 어디에 쓰이는지 연결해 볼 수 있었다.==

Transformer를 공부하면서는 Self-Attention에서 이해되지 않는 부분이 많았다. 특히 Q, K, V는 솔직히 제대로 이해하지 못한 채 넘어가려고 했다.

그때 [3Blue1BrownKR 유튜브 채널](https://www.youtube.com/@3Blue1BrownKR)의 그래픽을 활용한 설명이 많은 도움이 됐다. 글과 식으로만 봤을 때 잘 연결되지 않았던 내용이 시각적으로 보이니 훨씬 이해하기 쉬웠다. 그냥 넘길 뻔한 부분을 이해할 수 있게 설명해 주어서 감사했다.

이후에 좋은 그래픽카드를 얻게 된다면, 손실함수를 이용해 모델의 가중치를 조정하는 학습도 해보고 싶다. Config를 바꿨을 때 logits가 어떻게 달라지는지, 학습하면서 손실이 어디까지 낮아지는지도 비교해 보고 싶다.
