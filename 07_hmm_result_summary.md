# HMM Regime Model Result Summary
## Model: M1_sp500

---

### 1. 분석 목적

미국 시장의 월별 거시·시장 변수를 이용해 3-state Gaussian HMM으로 시장 레짐을 분류하고,
입력변수 조합별 분류 결과를 비교한다.

---

### 2. 사용 입력변수

- Model ID: `M1_sp500`
- Model Name: S&P500 only
- Features (1개): sp500_log_return

---

### 3. 전처리 방식

- 4개 데이터를 월별(Year-Month Period) 기준으로 inner join 병합
- BaaAaa_Spread는 일별 데이터를 월말 마지막 관측값으로 변환
- 결측치 포함 행 제거
- 각 모델별 입력변수에 대해 StandardScaler 적용 (in-sample 기준 fit, OOS는 transform만)

---

### 4. In-sample / OOS 기간

- In-sample  : 1990-01 ~ 2018-12  (347개월)
- OOS        : 2019-01 ~ latest   (88개월)
- 모델 학습은 in-sample에서만 수행
- OOS 구간에서는 모델 재학습 없이 학습된 파라미터로 Viterbi 분류만 수행

---

### 5. HMM 모델 설정

- n_components = 3
- covariance_type = diag
- n_iter = 1000
- n_inits = 20 (최적 log-likelihood 선택)
- random_state = 42
- 수렴 여부: True

---

### 6. 레짐별 경제적 해석 (In-sample 기준)

- **Bull**: 246 months (70.9%)  |  sp500_log_return=0.0145
- **Recovery**: 83 months (23.9%)  |  sp500_log_return=-0.0116
- **Bear**: 18 months (5.2%)  |  sp500_log_return=-0.0321

**레짐 라벨링 기준:**
sp500_log_return 평균을 기준으로 오름차순 정렬하여 Bear → Recovery → Bull 부여.
단일 변수 모델이므로 수익률 평균과 변동성만으로 제한적 라벨링.

---

### 7. 전이확률행렬

| From\\To | Bull | Recovery | Bear |
|---------|------|----------|------|
| Bull | 0.909 | 0.029 | 0.062 |
| Recovery | 0.118 | 0.882 | 0.000 |
| Bear | 0.608 | 0.392 | 0.000 |

**평균 지속기간:**
- Bull     : 11.0 months
- Recovery : 8.4 months
- Bear     : 1.0 months

---

### 8. AIC / BIC

| 항목 | 값 |
|------|----|
| Log Likelihood | -434.4246 |
| 파라미터 수 (k) | 14 |
| AIC | 896.8493 |
| BIC | 950.7398 |
| n (IS 관측치) | 347 |

AIC/BIC는 낮을수록 좋지만, 단독으로 최종 모델을 선택하면 안 됨.
레짐 해석 가능성, 관측 비중, 전이확률 안정성을 함께 고려해야 함.

---

### 9. OOS Viterbi 레짐 분류 결과 요약

- **Bull**: 66 months (75.0%)
- **Recovery**: 17 months (19.3%)
- **Bear**: 5 months (5.7%)

**주의:** OOS Viterbi 결과는 OOS 전체 관측값을 이용해 사후적으로 hidden state sequence를 추정한 것이다.
실시간 투자 예측과는 구분해야 한다.

---

### 10. 한계점

1. HMM state 번호는 임의적이며, sp500_log_return 평균 기준으로 라벨을 재부여함.
2. OOS Viterbi는 사후적 분류이므로 실시간 투자 신호로 사용 불가.
   - 실시간 투자전략에는 filtered posterior probability 또는 rolling/sequential updating이 필요.
3. Diagonal covariance 가정으로 변수 간 공분산을 무시함.
4. HMM은 변수들이 가우시안 분포를 따른다고 가정하지만, 금융 시계열은 fat tail을 가질 수 있음.
5. 단일 변수 모델은 레짐 식별력이 낮을 수 있으며, 경제적 해석이 제한적임.

---

### 11. 이후 확장 가능성

- Filtered posterior를 이용한 실시간 레짐 예측 및 투자전략 구성
- 레짐별 섹터 ETF 성과 분석 및 동적 포트폴리오 백테스트
- Full covariance 또는 tied covariance 모델과 비교
- Regime-switching GARCH 모델과 결합
- BIC 최적 n_states 탐색 (n=2,3,4 비교)
