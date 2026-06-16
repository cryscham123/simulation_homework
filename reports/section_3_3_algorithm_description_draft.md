# 3.3 Composite Rule — Algorithm 1 설명 (초안)

---

> **[연결 문장]**
> "… PSO 알고리즘 내에서 하나의 입자(Particle)는 6차원의 가중치 계수 벡터 $\mathbf{w} \in \mathbb{R}^6$으로 인코딩되며, 각 입자는 스스로 경험한 최적의 위치와 군집 전체가 발견한 최적의 위치를 기반으로 탐색 공간 내에서 속도와 위치 변수를 업데이트하며 최적 가중치를 추적한다."

---

## Algorithm 1 설명 본문

**초기화 단계(Line 1–9)**에서는 $N$개의 입자 위치를 탐색 공간 내에 고르게 분산시키기 위해, 각 입자의 초기 위치 $x_i$를 6차원 균등 Dirichlet 분포 $\text{Dir}(\mathbf{1})$에서 샘플링한다. 이를 통해 초기 군집은 단체(Simplex) 전반에 걸쳐 균등하게 분포되며, 특정 가중치 조합에 편향되지 않는 탐색 출발점을 보장한다. 초기 속도 $V_i$는 0으로 설정된다. 이후 각 입자의 개인 최적 위치($\text{Pbest}_i$)를 초기 위치로, 목적함수값 $F(\text{Pbest}_i)$를 해당 위치의 목적함수값으로 초기화하고, 군집 전체의 최적 위치 $\text{gbest}$는 개인 최적들 중 목적함수값이 최소인 입자로 설정한다.

**업데이트 단계(Line 10–24)**에서는 매 반복 $t$마다 다음의 속도-위치 갱신 규칙을 적용한다.

$$v_i \leftarrow \omega v_i + c_1 r_1 (\text{Pbest}_i - x_i) + c_2 r_2 (\text{gbest} - x_i)$$

$$x_i \leftarrow x_i + v_i$$

여기서 $\omega$는 관성 계수(inertia weight)로 이전 속도의 영향력을 조절하며, $c_1$과 $c_2$는 각각 인지적 계수(cognitive coefficient)와 사회적 계수(social coefficient)이다. $r_1, r_2 \sim U(0,1)^D$는 매 반복 독립적으로 샘플링되는 확률 벡터로, 탐색의 다양성을 유지한다.

위치 갱신 이후, 갱신된 위치 $x_i$가 6차원 단체(6-simplex) 위에 놓이도록 **단체 투영(Simplex Projection, Line 17)**을 적용한다. 구체적으로, 음수 성분을 0으로 클리핑한 뒤 각 차원의 값을 전체 합으로 나누어 정규화한다.

$$x_{i,d} \leftarrow \frac{\max(x_{i,d},\, 0)}{\sum_{d'} \max(x_{i,d'},\, 0)}$$

이 투영은 $\sum_d w_d = 1$이고 $w_d \geq 0$인 단체 제약을 항상 만족시키며, Composite Rule의 우선순위 점수 $\pi$가 가중치 비율에만 의존하고 절대적 크기에 무관하다는(scale-invariant) 성질을 자연스럽게 반영한다.

각 입자의 위치 $x_i$에 대한 목적함수 $F(x_i)$는 **표본평균근사(Sample Average Approximation, SAA)** 방식으로 평가한다. 동일한 공통난수(Common Random Numbers, CRN) 시드 집합 $\{s_1, \ldots, s_N\}$하에 $N$회의 시뮬레이션을 수행하여, Makespan과 평균 Q-Time 위반량(AQT)을 측정한다.

$$F(\mathbf{w}) = \hat{J}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{Mk_i(\mathbf{w})}{mk_{\text{ref}}} + \frac{\text{AQT}_i(\mathbf{w})}{qv_{\text{ref}}} \right]$$

여기서 $mk_{\text{ref}}$와 $qv_{\text{ref}}$는 각각 Makespan과 AQT의 정규화 기준값으로, 두 목적 사이의 스케일 차이를 보정하기 위해 기준 디스패칭 룰(SPTSSU, MIN\_QTIME)의 평균값으로부터 사전 산출된다. CRN을 통해 모든 입자와 반복에 걸쳐 동일한 확률적 환경이 적용되므로, 목적함수값의 차이가 가중치 벡터의 차이에서만 기인하도록 보장하여 탐색의 분산을 줄인다.

갱신된 위치가 개인 최적보다 우수한 경우 $\text{Pbest}_i$를 갱신하고(Line 20), 군집 전체 최적 $\text{gbest}$도 이에 맞게 갱신한다(Line 22). 이 과정을 최대 $T_{\max}$ 반복 수행한 뒤, 최종 군집 최적 $\mathbf{w}^* = \text{gbest}$를 출력한다(Line 25).
