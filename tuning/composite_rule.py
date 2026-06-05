"""
COMPOSITE rule 항 선별 + 가중치 최적화 (Stage 0 + Stage 1).

목적: grid_search.ipynb 의 4슬롯·격자 탐색 방식을 대체해
      ① 단일 항 grid (보조 자료)
      ② Shapley 누적 elbow curve
      ③ Shapley value (φ_i)
      ④ Pairwise interaction (φ_ij)
      를 한 번의 2^n-1 부분집합 평가에서 산출한다.

목적함수 (정규화 ON):
    J(w; S) = (1/N) Σ_i [ Mk_i / mk_ref + AQT_i / qv_ref ]
    where  AQT = QV_total / N_jobs,
           mk_ref, qv_ref ← data/normalization_ref.json (qv_ref 는 AQT 단위)

부분집합 가치함수:
    v(S) = min_{w ∈ simplex(|S|)} J(w; S)   ← PSO inner-loop
    v(∅) = 0   (Shapley baseline 관습)

이 모듈은 평가기·분석 함수만 제공한다. 노트북에서:
    from tuning.composite_rule import (
        load_normalization_ref, DynamicCompositeEvaluator,
        single_term_screening, shapley_value, pairwise_interaction,
        cumulative_J_curve,
    )
"""
import os
import json
import time
import random
from itertools import combinations
from math import factorial, comb

import numpy as np
import pandas as pd
import simpy

from utils import EventLogger
from simulation import Scheduler
from simulation.priority import (
    TERM_REGISTRY,
    available_terms,
)
from tuning.saa_evaluator import (
    _makespan,
    _qtime_violation,
    load_transition_risk_table,
    TRANSITION_RISK_CACHE_PATH,
)


# ===========================================================================
# 1) 정규화 기준 로드
# ===========================================================================
def load_normalization_ref(path):
    """data/normalization_ref.json → (mk_ref, qv_ref_aqt, raw_dict).

    json 의 qv_ref 는 QV_total(분) 이므로 AQT 단위로 환산하여 반환한다
    (qv_ref_aqt = qv_ref / n_jobs). J 식 내부에서 AQT 를 비교할 때 단위가 맞는다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        ref = json.load(f)
    mk_ref = float(ref['mk_ref'])
    qv_ref_total = float(ref['qv_ref'])
    n_jobs = int(ref.get('n_jobs', 60))
    if qv_ref_total <= 0 or n_jobs <= 0:
        raise ValueError(f"잘못된 normalization_ref: qv_ref={qv_ref_total}, n_jobs={n_jobs}")
    qv_ref_aqt = qv_ref_total / n_jobs
    return mk_ref, qv_ref_aqt, ref


# ===========================================================================
# 2) Dynamic-length COMPOSITE 평가기
# ===========================================================================
class DynamicCompositeEvaluator:
    """가변 길이 항 집합(|S|=1..n)을 지원하는 COMPOSITE rule SAA 평가기.

    saa_evaluator.SAAEvaluator 는 4슬롯 고정이라 |S|≥5 부분집합을 다룰 수 없다.
    이 평가기는 priority.composite_select 의 dynamic 모드
    (env var: COMPOSITE_TERMS, COMPOSITE_WEIGHTS) 를 사용해 임의 |S| 를 처리한다.

    목적함수: J = Mk/mk_ref + AQT/qv_ref  (가중치 (w1,w2)=(1,1), 정규화 ON)
    """

    def __init__(self, data, mk_ref, qv_ref_aqt,
                 base_seed=0,
                 down_active=True, pm_active=False,
                 pm_hazard_threshold=0.1,
                 transition_risk_path=None,
                 obj_weights=(1.0, 1.0)):
        self.data = data
        self.mk_ref = float(mk_ref)
        self.qv_ref = float(qv_ref_aqt)
        if self.mk_ref <= 0 or self.qv_ref <= 0:
            raise ValueError(f"mk_ref/qv_ref must be > 0 (got {self.mk_ref}, {self.qv_ref})")

        self.base_seed = int(base_seed)
        self.w1, self.w2 = obj_weights
        self.pm_hazard_threshold = pm_hazard_threshold
        self.n_jobs = int(len(data['jobs']))
        if self.n_jobs <= 0:
            raise ValueError("data['jobs'] 가 비어 있어 AQT 분모를 만들 수 없다.")

        os.environ['PM_ACTIVE'] = 'True' if pm_active else 'False'
        os.environ['DOWN_ACTIVE'] = 'True' if down_active else 'False'
        os.environ['PM_RULE'] = 'THRESHOLD'
        os.environ['TIME_UNIT'] = 'M'
        self.down_active = down_active
        self.pm_active = pm_active

        # transition 위험 테이블 캐시 로드 (saa_evaluator 가 미리 산출/저장한 것)
        path = transition_risk_path or TRANSITION_RISK_CACHE_PATH
        self.transition_risk = load_transition_risk_table(path)
        if not self.transition_risk:
            raise FileNotFoundError(
                f"transition_risk 캐시 없음: {path}\n"
                "먼저 SAAEvaluator 를 한 번 초기화해 캐시를 생성하라."
            )

    # ---- 단일 시뮬레이션 ----
    def _run_once(self, job_rule, seed, terms=None, weights=None):
        """terms/weights 가 주어지면 dynamic 모드, None 이면 legacy 또는 비-COMPOSITE."""
        random.seed(seed)
        env = simpy.Environment()
        logger = EventLogger(env)
        os.environ['JOB_RULE'] = job_rule

        if terms is not None and weights is not None:
            os.environ['COMPOSITE_TERMS'] = ','.join(terms)
            os.environ['COMPOSITE_WEIGHTS'] = ','.join(f'{float(w):.8f}' for w in weights)
        else:
            os.environ.pop('COMPOSITE_TERMS', None)
            os.environ.pop('COMPOSITE_WEIGHTS', None)

        scheduler = Scheduler(
            env=env, data=self.data, event_logger=logger,
            pm_hazard_threshold=self.pm_hazard_threshold,
        )
        env.run(until=scheduler.job_chk_process)
        return pd.DataFrame(logger.logs)

    def _seed_list(self, n_runs):
        return [self.base_seed + i for i in range(n_runs)]

    def _objective(self, mk, qv_total):
        aqt = qv_total / self.n_jobs
        return self.w1 * mk / self.mk_ref + self.w2 * aqt / self.qv_ref

    def _summarize(self, mks, qvs):
        js = np.asarray([self._objective(m, q) for m, q in zip(mks, qvs)], dtype=float)
        n = len(js)
        se = float(js.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
        return {
            'J_hat': float(js.mean()),
            'se': se,
            'makespan_mean': float(np.mean(mks)),
            'qtime_violation_mean': float(np.mean(qvs)),
            'aqt_mean': float(np.mean(qvs) / self.n_jobs),
            'J_samples': js,
            'n_runs': n,
        }

    # ---- 공개 API ----
    def evaluate(self, terms, weights, n_runs=10):
        """COMPOSITE(terms, weights) 의 SAA 추정.

        Args:
            terms: ['PT', 'SLACK', ...] (TERM_REGISTRY key, 1..n 개)
            weights: 같은 길이 실수 리스트. simplex 가정 (sum=1) 이 자연스럽지만
                     scale-invariant 라 sum≠1 도 동일 dispatch.
        """
        terms = [t.upper() for t in terms]
        weights = list(weights)
        if len(terms) != len(weights):
            raise ValueError(f"terms({len(terms)}) vs weights({len(weights)}) 길이 불일치")
        for t in terms:
            if t not in TERM_REGISTRY:
                raise KeyError(f"등록되지 않은 항: {t} (사용 가능: {available_terms()})")

        seeds = self._seed_list(n_runs)
        mks, qvs = [], []
        for s in seeds:
            df = self._run_once('COMPOSITE', s, terms=terms, weights=weights)
            mks.append(_makespan(df))
            qvs.append(_qtime_violation(df))
        return self._summarize(mks, qvs)

    def evaluate_rule(self, job_rule, n_runs=10):
        """baseline rule (FIFO/SPT/MIN_QTIME/...) 평가. 동일 CRN."""
        seeds = self._seed_list(n_runs)
        mks, qvs = [], []
        for s in seeds:
            df = self._run_once(job_rule, s, terms=None, weights=None)
            mks.append(_makespan(df))
            qvs.append(_qtime_violation(df))
        return self._summarize(mks, qvs)

    def summary(self):
        return {
            'mk_ref': self.mk_ref,
            'qv_ref_aqt': self.qv_ref,
            'obj_weights': (self.w1, self.w2),
            'n_jobs': self.n_jobs,
            'down_active': self.down_active,
            'pm_active': self.pm_active,
            'transition_risk_size': len(self.transition_risk),
        }


# ===========================================================================
# 3) Stage 0 — 단일 항 grid (보조 자료)
# ===========================================================================
def single_term_screening(evaluator, candidate_terms, n_runs=10, verbose=True):
    """각 단일 항을 단독 dispatch (weight=1.0) 했을 때의 J 측정.

    Shapley 결과와 대조해 "단독으로 강한 항" vs "결합으로 강한 항" 을 구분.
    elbow 의 선결 조건은 아니지만 보고서의 narrative 보조 자료로 유용.
    """
    rows = []
    t0 = time.time()
    for i, term in enumerate(candidate_terms):
        r = evaluator.evaluate([term], [1.0], n_runs=n_runs)
        rows.append({
            'term': term,
            'J_hat': r['J_hat'],
            'se': r['se'],
            'makespan_mean': r['makespan_mean'],
            'aqt_mean': r['aqt_mean'],
            'n_runs': r['n_runs'],
        })
        if verbose:
            print(f"  [{i+1}/{len(candidate_terms)}] {term:>16s}  "
                  f"J={r['J_hat']:.4f} ± {r['se']:.4f}  "
                  f"elapsed {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows).sort_values('J_hat').reset_index(drop=True)
    return df


# ===========================================================================
# 4) PSO on simplex (inner-loop for v(S))
# ===========================================================================
def pso_simplex(eval_fn, dim,
                swarm_size=8, n_iter=8,
                w_inertia=0.7, c_cog=1.5, c_soc=1.5,
                seed=0, verbose=False):
    """sum=1 simplex 위의 minimize PSO.

    각 particle 위치 x ∈ R^dim, 평가 시 x/sum(x) 로 정규화 (scale-invariance).
    초기 위치 = Dirichlet(1,...,1) 균등 simplex 샘플.

    예산: (swarm_size) × (n_iter + 1) 회 eval_fn 호출.

    Returns:
        dict {'w*', 'J*', 'history'}
    """
    if dim <= 0:
        raise ValueError("dim must be >= 1")
    rng = np.random.default_rng(seed)

    pos = rng.dirichlet(np.ones(dim), size=swarm_size)
    vel = np.zeros_like(pos)

    p_best_pos = pos.copy()
    p_best_J = np.array([eval_fn(p) for p in pos])

    g_idx = int(np.argmin(p_best_J))
    g_best_pos = p_best_pos[g_idx].copy()
    g_best_J = float(p_best_J[g_idx])

    history = [(0, g_best_J)]

    for it in range(1, n_iter + 1):
        r1 = rng.random(size=(swarm_size, dim))
        r2 = rng.random(size=(swarm_size, dim))
        vel = (w_inertia * vel
               + c_cog * r1 * (p_best_pos - pos)
               + c_soc * r2 * (g_best_pos - pos))
        pos = pos + vel
        pos = np.clip(pos, 0.0, None)
        sums = pos.sum(axis=1, keepdims=True)
        sums[sums < 1e-12] = 1.0
        pos = pos / sums

        Js = np.array([eval_fn(p) for p in pos])
        improved = Js < p_best_J
        p_best_pos[improved] = pos[improved]
        p_best_J[improved] = Js[improved]

        g_idx = int(np.argmin(p_best_J))
        if p_best_J[g_idx] < g_best_J:
            g_best_pos = p_best_pos[g_idx].copy()
            g_best_J = float(p_best_J[g_idx])

        history.append((it, g_best_J))
        if verbose:
            print(f"    PSO iter {it}/{n_iter}  g*={g_best_J:.5f}")

    return {'w*': g_best_pos, 'J*': g_best_J, 'history': history}


def value_function(evaluator, terms_S, n_runs, pso_kwargs):
    """v(S) = min_{w ∈ simplex(|S|)} J(w; S).

    |S|=0 → FIFO J (information-less baseline).
            J=0 sentinel 을 쓰면 minimization 에서 v(∅) 가 글로벌 최적처럼
            취급되어 모든 marginal 이 음수 편향되므로, 실제 "아무 우선순위
            정보도 없는 dispatch" 대용으로 FIFO 의 SAA J 를 baseline 으로
            사용한다. (동일 CRN seed.)
    |S|=1 → PSO 불필요 (w=1.0, scale-invariant)
    |S|≥2 → PSO inner-loop
    """
    k = len(terms_S)
    if k == 0:
        r = evaluator.evaluate_rule('FIFO', n_runs=n_runs)
        return {'J*': r['J_hat'], 'w*': np.array([])}
    if k == 1:
        r = evaluator.evaluate(list(terms_S), [1.0], n_runs=n_runs)
        return {'J*': r['J_hat'], 'w*': np.array([1.0])}

    terms_list = list(terms_S)

    def _eval(w):
        return evaluator.evaluate(terms_list, w.tolist(), n_runs=n_runs)['J_hat']

    out = pso_simplex(_eval, dim=k, **pso_kwargs)
    return {'J*': out['J*'], 'w*': out['w*']}


# ===========================================================================
# 5) Shapley value (exact, 2^n - 1 subsets)
# ===========================================================================
def shapley_value(evaluator, candidate_terms,
                  n_runs=10,
                  pso_kwargs=None,
                  value_cache=None,
                  verbose=True):
    """Exact Shapley value via 2^n 부분집합 평가.

    Baseline 규약:
        v(∅) = FIFO J (information-less dispatch). J=0 sentinel 은 minimization
        에서 모든 marginal 을 음수로 끌어내려 φ 부호 해석을 깨므로, 실제로
        "어떤 우선순위 항도 안 쓰는 dispatch" 에 해당하는 FIFO 의 SAA J 를
        baseline 으로 사용한다.

    최소화 목적함수이므로 marginal contribution 부호 규약:
        Δ_t(S) := v(S) - v(S ∪ {t})    (t 가 들어가면 J 가 얼마나 감소했나)
        φ_t = Σ_{S ⊆ N\\{t}} (|S|! (n-|S|-1)! / n!) · Δ_t(S)
    → φ_t > 0  ⇔  t 가 평균적으로 J 를 줄인다 (= 유익).
       (FIFO baseline 하에서 Σ φ_t = v(∅) - v(N) = J_FIFO - J_full > 0.)

    Args:
        evaluator: DynamicCompositeEvaluator
        candidate_terms: 분석 대상 항 리스트 (n ≤ 8 권장; n=6 이면 64 subset)
        n_runs: 부분집합 평가용 SAA replication 수
        pso_kwargs: PSO 인자 dict. None 이면 swarm=8/n_iter=8.
        value_cache: 사전 계산된 v(S) 캐시 (frozenset(terms) → {'J*','w*'})
        verbose: 진행 출력

    Returns:
        phi: dict term → φ_t (양수일수록 J 감소 기여)
        value_cache: 갱신된 v(S) 캐시
        meta: {'n_subsets', 'elapsed_sec', 'pso_kwargs', 'n_runs'}
    """
    if pso_kwargs is None:
        pso_kwargs = dict(swarm_size=8, n_iter=8, seed=0)
    if value_cache is None:
        value_cache = {}

    n = len(candidate_terms)
    if n < 2:
        raise ValueError(f"Shapley 는 n≥2 필요 (받은 n={n})")
    if n > 8 and verbose:
        print(f"⚠ n={n} 은 2^n={2**n} 부분집합 → 비용이 빠르게 커진다.")

    terms = list(candidate_terms)

    # 구버전 (v(∅)=0 sentinel) 캐시 무효화 → FIFO baseline 으로 재평가 유도
    _empty = frozenset()
    if _empty in value_cache and value_cache[_empty].get('J*', None) == 0.0:
        if verbose:
            print("  [cache] 구버전 v(∅)=0 sentinel 감지 → FIFO baseline 으로 재평가")
        del value_cache[_empty]

    all_subsets = [frozenset()]
    for k in range(1, n + 1):
        for combo in combinations(terms, k):
            all_subsets.append(frozenset(combo))

    t0 = time.time()
    for idx, S in enumerate(all_subsets):
        if S in value_cache:
            continue
        out = value_function(evaluator, sorted(S), n_runs, pso_kwargs)
        value_cache[S] = out
        if verbose:
            print(f"  [{idx+1}/{len(all_subsets)}] |S|={len(S)}  "
                  f"v(S)={out['J*']:.5f}  elapsed {time.time()-t0:.1f}s")

    phi = {t: 0.0 for t in terms}
    for t in terms:
        for S in all_subsets:
            if t in S:
                continue
            S_with = frozenset(S | {t})
            v_without = value_cache[S]['J*']
            v_with = value_cache[S_with]['J*']
            marginal = v_without - v_with
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            phi[t] += w * marginal

    meta = {
        'n_subsets': len(all_subsets),
        'elapsed_sec': time.time() - t0,
        'pso_kwargs': dict(pso_kwargs),
        'n_runs': n_runs,
        'candidate_terms': terms,
    }
    return phi, value_cache, meta


# ===========================================================================
# 6) Shapley interaction index (pairwise, exact)
# ===========================================================================
def pairwise_interaction(value_cache, candidate_terms):
    """Shapley interaction index (Grabisch, 1997) — pairwise.

        I_{ij} = Σ_{S ⊆ N\\{i,j}}  [ |S|! (n-|S|-2)! / (n-1)! ] · δ_{ij}(S)
        δ_{ij}(S) = v(S ∪ {i,j}) - v(S ∪ {i}) - v(S ∪ {j}) + v(S)

    부호 규약 (최소화):
        δ < 0  ⇔  i 와 j 가 함께 들어가면 J 가 합의 단독 효과보다 더 줄어든다
                  ⇒ **시너지** (보완재)
        δ > 0  ⇔  함께 들어가면 효과가 상쇄 ⇒ **상충** (대체재)

    UI 에서는 보통 -I_{ij} 를 "시너지 강도" 로 시각화하기도 한다.
    """
    terms = list(candidate_terms)
    n = len(terms)
    if n < 2:
        raise ValueError("interaction 은 n≥2 필요")

    out = {}
    for ai in range(n):
        for aj in range(ai + 1, n):
            i, j = terms[ai], terms[aj]
            rest = [t for t in terms if t != i and t != j]
            total = 0.0
            for k in range(0, n - 1):
                for S in combinations(rest, k):
                    Sset = frozenset(S)
                    v_S = value_cache[Sset]['J*']
                    v_Si = value_cache[Sset | {i}]['J*']
                    v_Sj = value_cache[Sset | {j}]['J*']
                    v_Sij = value_cache[Sset | {i, j}]['J*']
                    delta = v_Sij - v_Si - v_Sj + v_S
                    w = factorial(k) * factorial(n - k - 2) / factorial(n - 1)
                    total += w * delta
            out[(i, j)] = total
    return out


# ===========================================================================
# 7) Cumulative J curve (Shapley 상위 k 누적 → elbow)
# ===========================================================================
def cumulative_J_curve(evaluator, phi, value_cache,
                       n_runs=10, pso_kwargs=None, verbose=True):
    """Shapley 상위 k 개 항만으로 v(top-k) → elbow 곡선.

    φ 내림차순으로 항을 누적 추가:
        k=1 → 가장 기여 큰 단일 항
        k=2 → 상위 2 개
        ...
        k=n → 전체 (full set 의 v)

    value_cache 에 이미 그 부분집합이 있으면 재사용 (Shapley 단계에서 다 채워짐).
    """
    if pso_kwargs is None:
        pso_kwargs = dict(swarm_size=8, n_iter=8, seed=0)

    ranked = sorted(phi.items(), key=lambda x: -x[1])
    rows = []
    for k in range(1, len(ranked) + 1):
        top_k = [t for t, _ in ranked[:k]]
        S = frozenset(top_k)
        if S in value_cache:
            J_star = value_cache[S]['J*']
            w_star = value_cache[S]['w*']
        else:
            out = value_function(evaluator, top_k, n_runs, pso_kwargs)
            J_star = out['J*']
            w_star = out['w*']
            value_cache[S] = out
        rows.append({
            'k': k,
            'terms': tuple(top_k),
            'J*': J_star,
            'w*': tuple(float(x) for x in w_star) if len(w_star) else (),
        })
        if verbose:
            print(f"  k={k:2d}  J*={J_star:.5f}  terms={top_k}")
    return pd.DataFrame(rows)


# ===========================================================================
# 8) Cost estimator (사전 비용 견적)
# ===========================================================================
def estimate_cost(n_terms, n_runs=10, pso_kwargs=None, sec_per_sim=0.85):
    """Shapley 전체 비용 견적.

    각 부분집합 비용:
        |S|=0 → 0
        |S|=1 → n_runs sim
        |S|≥2 → swarm × (n_iter+1) × n_runs sim

    sec_per_sim 은 grid_search.ipynb 의 696s/840sim ≈ 0.83 → 0.85 default.
    """
    if pso_kwargs is None:
        pso_kwargs = dict(swarm_size=8, n_iter=8)
    swarm = pso_kwargs.get('swarm_size', 8)
    n_iter = pso_kwargs.get('n_iter', 8)
    pso_evals = swarm * (n_iter + 1)

    total_sims = 0
    breakdown = []
    for k in range(0, n_terms + 1):
        n_subsets_k = comb(n_terms, k)
        if k == 0:
            sims_k = 0
        elif k == 1:
            sims_k = n_subsets_k * n_runs
        else:
            sims_k = n_subsets_k * pso_evals * n_runs
        total_sims += sims_k
        breakdown.append({'|S|': k, 'n_subsets': n_subsets_k, 'sims': sims_k})

    return {
        'n_terms': n_terms,
        'n_subsets_total': 2 ** n_terms,
        'pso_evals_per_subset': pso_evals,
        'n_runs': n_runs,
        'sims_total': total_sims,
        'sec_per_sim': sec_per_sim,
        'est_seconds': total_sims * sec_per_sim,
        'est_hours': total_sims * sec_per_sim / 3600.0,
        'breakdown': pd.DataFrame(breakdown),
    }
