# 3.4 PM Rule

## 3.4.1 예방 보전 정책 개요

본 연구의 시뮬레이션은 설비 고장이 스케쥴링 성능에 미치는 영향을 현실적으로 반영하기 위해 예방 보전(Preventive Maintenance, PM) 정책을 명시적으로 모델링한다. PM 정책의 핵심 문제는 **언제 PM을 수행할 것인가**, 즉 PM 발동 시각(PM trigger time) $T_{PM}$을 결정하는 것이다. $T_{PM}$ 이전에 고장이 발생하면 PM은 취소되고 사후 수리(Corrective Maintenance, CM)가 수행되며, 반대로 $T_{PM}$에 도달하면 설비를 선제적으로 점검함으로써 예상치 못한 중단을 방지한다.

본 연구에서는 세 가지 PM Rule을 구현하여 비교한다: **Threshold Rule**, **Age Replacement Rule**, **MTTF Rule**. 모든 Rule은 Weibull 분포 기반 고장 모델을 공유하며, 환경 변수 `PM_RULE`로 실험 시 선택한다.

---

## 3.4.2 고장 모델: Weibull 분포

설비 고장 시각은 2-모수 Weibull 분포를 따른다고 가정한다. 누적 고장 확률(CDF)과 신뢰도 함수(Survival Function)는 각각 다음과 같다.

$$F(t) = 1 - \exp\!\left[-\left(\frac{t}{\lambda}\right)^k\right], \quad R(t) = \exp\!\left[-\left(\frac{t}{\lambda}\right)^k\right]$$

여기서 $k$는 형상 모수(shape parameter), $\lambda$는 척도 모수(scale parameter)이다. 본 실험에서는 전 설비에 동일하게 $k = 3.0$, $\lambda = 11{,}000$분을 적용하였다. $k > 1$이므로 고장률(hazard rate)은 시간에 따라 단조 증가하며, 이는 노화(aging) 특성을 갖는 설비에 적합한 분포이다.

Weibull 분포의 평균 고장 간격(MTTF)은 다음과 같이 계산된다.

$$\text{MTTF} = \lambda \cdot \Gamma\!\left(1 + \frac{1}{k}\right)$$

$k = 3.0$, $\lambda = 11{,}000$을 대입하면 $\text{MTTF} \approx 9{,}823$분(약 163.7시간)이다.

시뮬레이션에서 개별 설비의 고장 발생 시각은 역변환법(Inverse Transform Sampling)으로 샘플링된다.

$$t_{\text{failure}} = \lambda \cdot (-\ln U)^{1/k}, \quad U \sim \text{Uniform}(0, 1)$$

---

## 3.4.3 PM Rule 1: Threshold Rule

### 정의

Threshold Rule은 누적 고장 확률 $F(t)$가 사전에 정의된 임계값 $p$에 도달하는 시점을 PM 발동 시각으로 설정한다.

$$T_{PM}^{\text{THR}} = \lambda \cdot \left(-\ln(1 - p)\right)^{1/k}$$

### 해석

이 Rule은 직관적인 위험 기반 접근법으로, 설비가 고장날 확률이 $p$에 달하기 전에 선제적으로 점검을 수행한다는 논리다. 임계값 $p$는 환경 변수 `PM_HAZARD_THRESHOLD`로 설정하며, 본 실험에서는 $p = 0.1$ (10%)을 기본값으로 사용하였다.

$k = 3.0$, $\lambda = 11{,}000$, $p = 0.1$을 대입하면:

$$T_{PM}^{\text{THR}} = 11{,}000 \cdot (-\ln 0.9)^{1/3} \approx 5{,}195 \text{ 분 (약 86.6시간)}$$

즉 설비가 마지막 PM 또는 수리 이후 약 86.6시간 가동되면 PM이 발동된다.

---

## 3.4.4 PM Rule 2: Age Replacement Rule

### 정의

Age Replacement Rule은 Barlow and Proschan (1965)의 연령 교체 모델에 기반한다. PM 비용 $c_{pm}$과 고장 수리 비용 $c_{r}$ ($c_r > c_{pm}$)이 주어졌을 때, 단위 시간당 기대 비용을 최소화하는 최적 PM 주기 $T^*$를 구한다.

$$T^* = \arg\min_{T > 0} \; g(T), \quad g(T) = \frac{c_{pm} \cdot R(T) + c_{r} \cdot F(T)}{\int_0^T R(u)\, du}$$

분모 $\int_0^T R(u)\, du$는 한 교체 사이클의 기대 가동 시간이다. $k \leq 1$이면 고장률이 감소하거나 일정하여 PM이 무의미하므로($T^* = \infty$), $k > 1$인 경우에만 적용된다. 마찬가지로 $c_r \leq c_{pm}$이면 고장을 기다리는 편이 경제적이므로 PM을 수행하지 않는다.

### 수치 결과

본 실험 데이터에서 $c_{pm} = 80$분(PM 수행 시간), $c_r = 220$분(수리 시간), $k = 3.0$, $\lambda = 11{,}000$분을 적용하면:

$$T^* \approx 7{,}332 \text{ 분 (약 122.2시간)}$$

$T^*$는 MTTF(9,823분)의 약 74.6%에 해당하며, Threshold Rule(5,195분)보다 늦은 시점에 PM이 발동된다. 이는 Age Replacement Rule이 수리 비용 대비 PM 비용의 비율을 명시적으로 고려하기 때문으로, 불필요하게 이른 PM을 방지하는 경향이 있다.

수치 최적화는 SciPy의 `minimize_scalar` (bounded 방법)를 사용하며, 탐색 범위는 $[0.05 \cdot \text{MTTF},\; 5.0 \cdot \text{MTTF}]$로 설정하였다. 동일한 $(k, \lambda, c_r, c_{pm})$ 조합에 대해 반복 계산을 피하기 위해 `lru_cache`로 결과를 캐싱한다.

---

## 3.4.5 PM Rule 3: MTTF Rule

### 정의

MTTF Rule은 설비 그룹(machine group)의 공정 위험도(risk level)에 따라 차등 PM 주기를 부여한다.

$$T_{PM}^{\text{MTTF}} = \alpha_g \cdot \text{MTTF}, \quad \text{MTTF} = \lambda \cdot \Gamma\!\left(1 + \frac{1}{k}\right)$$

여기서 $\alpha_g$는 그룹 $g$의 위험도에 대응하는 계수이다. 위험도와 계수 매핑은 다음과 같다.

| 위험도 | 계수 $\alpha$ | 대응 그룹 |
|:------:|:-------------:|:---------:|
| High   | 0.3           | G3, G4    |
| Medium | 0.5           | G1, G5    |
| Low    | 0.7           | G2        |

### 수치 결과

$\text{MTTF} \approx 9{,}823$분을 대입하면 각 그룹의 PM 주기는 다음과 같다.

| 그룹 | 위험도 | PM 주기 (분) | PM 주기 (시간) |
|:----:|:------:|:------------:|:--------------:|
| G3, G4 | High   | 2,947 | 49.1 |
| G1, G5 | Medium | 4,911 | 81.9 |
| G2     | Low    | 6,876 | 114.6 |

MTTF Rule은 QTime 위반 위험이 높은 전환(Transition)이 집중된 그룹에 더 잦은 PM을 부여함으로써, 설비 가용성과 QTime 준수율을 동시에 관리하는 전략이다. 본 데이터에서 G3과 G4는 QTime 제약이 있는 전환(G1→G3, G1→G4, G4→G1)에 직접 관여하는 그룹으로, High 위험도로 분류되었다.

---

## 3.4.6 시뮬레이션 내 PM 구현

### PM과 고장 프로세스의 상호작용

시뮬레이션(SimPy 기반)에서 각 설비는 시작 시점에 **고장 프로세스**(down)와 **PM 프로세스**(PM)를 병렬로 실행한다.

- 고장 프로세스: Weibull 샘플링 시간 $t_{\text{failure}}$ 후 설비 중단 → CM 수행(소요 시간 $c_r$)
- PM 프로세스: $T_{PM}$ 경과 후 PM 발동 → 설비 선제 점검(소요 시간 $c_{pm}$)

두 프로세스 중 먼저 발동된 쪽이 상대 프로세스를 인터럽트(interrupt)한다.

$$\text{실제 수행} = \begin{cases} \text{PM} & (T_{PM} < t_{\text{failure}}) \\ \text{CM} & (t_{\text{failure}} \leq T_{PM}) \end{cases}$$

### 우선순위 및 선점(Preemption)

설비 자원은 SimPy `PreemptiveResource`로 모델링된다. 수리(CM)는 `preempt=True`로 작업 중인 Job을 즉시 선점하는 반면, PM은 `preempt=False`로 현재 Job 처리가 완료된 후 유휴(IDLE) 상태에서 수행된다. 이는 PM이 계획된 활동이므로 Job 중단 없이 유휴 시간을 활용한다는 현실적 가정을 반영한다.

| 이벤트 | 우선순위 | 선점 여부 |
|:------:|:--------:|:---------:|
| CM(고장 수리) | -1 (높음) | True (Job 중단) |
| PM | -1 (높음) | False (Job 완료 후) |
| Job 처리 | 0 (낮음) | — |

### PM 완료 후 리셋

PM 또는 CM 완료 후 설비는 `IDLE` 상태로 복귀하며, 이전 Job 타입 정보(`last_job_type`)가 초기화된다. 이후 새로운 고장 프로세스와 PM 프로세스가 재시작된다(age-based 모델).

---

## 3.4.7 PM Rule별 특성 비교

| | Threshold | Age Replacement | MTTF |
|:--|:---------:|:---------------:|:----:|
| **PM 주기 결정 기준** | 고장 확률 임계값 | 기대 비용 최소화 | 위험도 기반 MTTF 비율 |
| **설비 그룹 차별화** | ✗ | ✗ | ✓ |
| **비용 정보 활용** | ✗ | ✓ ($c_{pm}$, $c_r$) | ✗ |
| **해석 용이성** | 높음 | 낮음 | 중간 |
| **본 실험 PM 시각** | 5,195분 | 7,332분 | 2,947~6,876분 |

Threshold Rule은 단순하지만 비용 구조를 고려하지 않는다. Age Replacement Rule은 경제적으로 최적화된 주기를 제공하나 그룹별 차이를 반영하지 못한다. MTTF Rule은 QTime 위험이 높은 그룹에 집중적 PM을 배분하는 도메인 지식 기반 접근법으로, 세 Rule 중 스케쥴링 목적함수(AQT)와 가장 직접적으로 연계된다.
