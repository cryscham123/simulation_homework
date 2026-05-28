"""
SAA(Sample Average Approximation) 기반 COMPOSITE rule 파라미터 평가 모듈.

목적:
  파라미터 벡터 w = (alpha, beta, gamma, delta)에 대해
    J_hat(w) = (1/N) Σ_i [ w1 · Makespan_i(w)/mk_ref + w2 · QV_i(w)/qv_ref ]
  를 추정한다.
"""
import os
import json
import random
import simpy
import numpy as np
import pandas as pd

from utils import EventLogger
from simulation import Scheduler
from simulation.priority import set_transition_risk_table, get_transition_risk_table

# transition 위험 테이블 캐시 파일.
# evaluator init 시 자동 저장 → result.ipynb 등 외부 노트북이 로드해
# 같은 테이블로 COMPOSITE를 재현할 수 있게 한다.
TRANSITION_RISK_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), 'transition_risk.json'
)


def load_transition_risk_table(path=TRANSITION_RISK_CACHE_PATH):
    """캐시 파일에서 transition 위험 테이블을 로드해 priority 모듈에 주입한다.

    grid_search 등에서 SAAEvaluator가 한 번 산출해 저장해 둔 테이블을
    외부(result.ipynb)에서 그대로 사용하기 위한 헬퍼다. 파일이 없으면
    빈 테이블이 유지된다(c_transition 항 무효).
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        table = json.load(f)
    set_transition_risk_table(table)
    return table


def _qtime_violation(log_df):
    """log DataFrame에서 qtime 위반 시간 총합을 계산한다."""
    qv = log_df[log_df['event'] == 'qtime_over']
    if len(qv) == 0:
        return 0.0
    return float((qv['finish'] - qv['start']).sum())


def _makespan(log_df):
    """log DataFrame에서 makespan(마지막 job 완료 시각)을 계산한다."""
    jl = log_df[log_df['resource'] == 'job']
    return float(jl.groupby('id')['finish'].max().max())


class SAAEvaluator:
    """COMPOSITE rule 파라미터에 대한 SAA 평가기."""

    PARAM_NAMES = ('alpha', 'beta', 'gamma', 'delta')

    def __init__(self, data,
                 n_ref=20,
                 obj_weights=(1.0, 1.0),
                 pm_hazard_threshold=0.1,
                 base_seed=0,
                 down_active=False,
                 pm_active=False,
                 mk_ref=None,
                 qv_ref=None):
        """
        Args:
            data: DataLoader.load_all_data() 결과
            n_ref: mk_ref, qv_ref 산출용 replication 수
            obj_weights: (w1, w2) — Makespan 항과 QV 항의 명시적 가중치.
                         기본 (1.0, 1.0) = 사용자가 의도한 1:1.
            pm_hazard_threshold: Scheduler에 넘길 PM 임계값 (PM 비활성 시 무영향)
            base_seed: seed_list 생성 기준값. seed_list[i] = base_seed + i
            mk_ref, qv_ref: 외부에서 산출한 정규화 기준값. 둘 다 주면
                내부 _compute_reference()를 건너뛴다.
                result_compare.ipynb가 6개 baseline rule × 30 run의 산술평균으로
                산출해 data/normalization_ref.json에 저장한 값을 그대로 받기 위함.
        """
        self.data = data
        self.pm_hazard_threshold = pm_hazard_threshold
        self.base_seed = base_seed
        self.w1, self.w2 = obj_weights
        self.n_ref = n_ref

        # DOWN/PM 활성 여부는 호출자가 결정한다(기본은 비활성으로 결정론 시나리오).
        # DOWN_ACTIVE=True면 Machine.down 프로세스가 활성화되어 같은 w에서도 seed별로
        # 다른 J가 나오고, J_samples 분산이 생겨 t-test와 SAA bias 관찰이 의미를 갖는다.
        os.environ['PM_ACTIVE'] = 'True' if pm_active else 'False'
        os.environ['DOWN_ACTIVE'] = 'True' if down_active else 'False'
        os.environ['PM_RULE'] = 'THRESHOLD'
        # 단위 안정화: evaluator 내부 계산은 분 단위로 통일. mk_ref/qv_ref와
        # J_hat 스케일이 .env의 TIME_UNIT에 흔들리지 않게 한다.
        os.environ['TIME_UNIT'] = 'M'
        self.down_active = down_active
        self.pm_active = pm_active

        # 1) transition 위험 가중치 테이블을 baseline 측정으로 산출·주입
        self.transition_risk = self._compute_transition_risk()
        set_transition_risk_table(self.transition_risk)
        # 외부 노트북(result.ipynb 등)이 같은 테이블로 COMPOSITE를 재현할 수 있게
        # 캐시 파일로 저장. load_transition_risk_table()로 로드한다.
        try:
            with open(TRANSITION_RISK_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.transition_risk, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        # 2) 목적함수 정규화 기준(mk_ref, qv_ref) 산출
        #    외부에서 mk_ref, qv_ref를 모두 주입하면 내부 산출을 건너뛴다.
        #    (result_compare.ipynb가 6개 rule × 30 run 산술평균으로 산출한 값을
        #     data/normalization_ref.json에서 로드해 넘기는 경로 지원)
        if mk_ref is not None and qv_ref is not None:
            self.mk_ref = float(mk_ref)
            self.qv_ref = float(qv_ref) if qv_ref > 0 else 1.0
            self._ref_source = 'external'
        else:
            self.mk_ref, self.qv_ref = self._compute_reference()
            self._ref_source = 'internal'

    # ---- 내부: 단일 시뮬레이션 ----
    def _run_once(self, job_rule, seed):
        """
        주어진 rule과 seed로 시뮬레이션 1회 실행 → log DataFrame.
        seed는 CRN을 위해 호출자가 결정론적으로 공급한다.
        """
        random.seed(seed)
        env = simpy.Environment()
        logger = EventLogger(env)
        os.environ['JOB_RULE'] = job_rule
        scheduler = Scheduler(
            env=env, data=self.data, event_logger=logger,
            pm_hazard_threshold=self.pm_hazard_threshold,
        )
        env.run(until=scheduler.job_chk_process)
        return pd.DataFrame(logger.logs)

    def _seed_list(self, n_runs):
        """replication별 고정 seed 리스트. CRN의 핵심."""
        return [self.base_seed + i for i in range(n_runs)]

    # ---- transition 위험 가중치 테이블 ----
    def _compute_transition_risk(self):
        """
        데이터 기반 결정론적 transition 위험도.

          risk_raw[A->B] = (1 / qtime_limit[A->B]) × frequency[A->B]
            - 1/qtime_limit: 제약이 짧을수록 위험
            - frequency      : job mix 안에서 A->B가 자주 일어날수록 위험

        최댓값으로 정규화해 [0,1] 위험 가중치 테이블을 만든다. policy 독립적이고
        stochastic noise도 없어 reproducible하다. breakdown 같은 불확실성 효과는
        runtime의 slack(β)/CR(α) 항이 잡으므로 c_transition은 선험적 위험만 담는다.

        qtime_limit이 inf 또는 0 이하인 transition은 '제약 없음'으로 보고 테이블에서
        제외 (c_transition=0). qtime_limit은 transition별 평균을 대표값으로 사용한다.
        """
        ops = self.data['operations'].sort_values(['job_id', 'op_seq']).copy()
        ops['prev_group'] = ops.groupby('job_id')['op_group'].shift(1)
        ops = ops.dropna(subset=['prev_group'])
        ops['transition'] = ops['prev_group'] + '->' + ops['op_group']

        qt = ops['qtime'].astype(float).copy()
        qt[qt <= 0] = float('inf')
        ops['qtime_eff'] = qt

        total = len(ops)
        if total == 0:
            return {}

        raw = {}
        for trans, grp in ops.groupby('transition'):
            qt_mean = float(grp['qtime_eff'].mean())
            if not np.isfinite(qt_mean) or qt_mean <= 0:
                continue
            freq = len(grp) / total
            raw[trans] = (1.0 / qt_mean) * freq

        if not raw:
            return {}
        peak = max(raw.values())
        if peak <= 0:
            return {}
        return {t: v / peak for t, v in raw.items()}

    # ---- 목적함수 정규화 기준 ----
    def _compute_reference(self):
        """
        utopia-point 정규화: 각 metric별로 가장 유리한 단일 rule의 평균을 기준값으로 삼는다.
          mk_ref = mean(makespan of SPTSSU runs)   # Makespan에 가장 유리한 rule
          qv_ref = mean(QV       of MIN_QTIME runs) # QV에 가장 유리한 rule

        의미: J=1.0이면 COMPOSITE가 두 metric 모두에서 단일 rule best와 동등.
              J<1.0이면 COMPOSITE가 두 축의 utopia point보다 우월 (= trade-off에서 유의미한 개선).
        qv_ref=0이면(위반 전무) 1.0으로 대체해 0 나눗셈을 막는다.
        """
        seeds = self._seed_list(self.n_ref)
        mks = [_makespan(self._run_once('SPTSSU', s)) for s in seeds]
        qvs = [_qtime_violation(self._run_once('MIN_QTIME', s)) for s in seeds]
        mk_ref = float(np.mean(mks))
        qv_ref = float(np.mean(qvs))
        if qv_ref <= 0:
            qv_ref = 1.0
        return mk_ref, qv_ref

    # ---- 파라미터 적용 ----
    def _apply_params(self, w):
        """w = (alpha, beta, gamma, delta)를 COMPOSITE_* 환경변수에 반영한다."""
        if len(w) != len(self.PARAM_NAMES):
            raise ValueError(
                f"w는 {len(self.PARAM_NAMES)}차원이어야 한다 (alpha,beta,gamma,delta). "
                f"받은 길이: {len(w)}"
            )
        for name, val in zip(self.PARAM_NAMES, w):
            os.environ[f'COMPOSITE_{name.upper()}'] = str(float(val))

    # ---- 목적함수 ----
    def _objective(self, makespan, qtime_violation):
        """
        단일 replication의 목적함수 값.
        J_i = w1 · (Makespan/mk_ref) + w2 · (QV/qv_ref)
        """
        return (self.w1 * makespan / self.mk_ref
                + self.w2 * qtime_violation / self.qv_ref)

    def _summarize(self, mks, qvs):
        """raw 결과 리스트 → J 통계 dict."""
        js = np.asarray([self._objective(m, q) for m, q in zip(mks, qvs)],
                        dtype=float)
        n = len(js)
        se = float(js.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
        return {
            'J_hat': float(js.mean()),
            'se': se,
            'makespan_mean': float(np.mean(mks)),
            'qtime_violation_mean': float(np.mean(qvs)),
            'J_samples': js,
            'n_runs': n,
        }

    # ---- 공개 API ----
    def evaluate(self, w, n_runs=20):
        """
        파라미터 w에 대한 J_hat(w)를 SAA로 추정한다.

        Args:
            w: (alpha, beta, gamma, delta)
            n_runs: replication 수 N

        Returns:
            dict: J_hat, se, makespan_mean, qtime_violation_mean, J_samples, n_runs
        """
        self._apply_params(w)
        seeds = self._seed_list(n_runs)
        mks, qvs = [], []
        for s in seeds:
            log_df = self._run_once('COMPOSITE', s)
            mks.append(_makespan(log_df))
            qvs.append(_qtime_violation(log_df))
        return self._summarize(mks, qvs)

    def evaluate_rule(self, job_rule, n_runs=20):
        """
        기존 rule(MIN_QTIME 등)을 동일 목적함수/CRN으로 평가.
        COMPOSITE 최적 w와 기존 rule을 같은 척도(J_hat)로 비교할 때 사용한다.
        """
        seeds = self._seed_list(n_runs)
        mks, qvs = [], []
        for s in seeds:
            log_df = self._run_once(job_rule, s)
            mks.append(_makespan(log_df))
            qvs.append(_qtime_violation(log_df))
        return self._summarize(mks, qvs)

    def summary(self):
        """초기화 결과(정규화 기준, transition 위험 테이블) 요약 dict."""
        return {
            'mk_ref': self.mk_ref,
            'qv_ref': self.qv_ref,
            'obj_weights': (self.w1, self.w2),
            'transition_risk': dict(self.transition_risk),
        }
