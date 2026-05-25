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


def _build_op_transition_map(data):
    """op_id -> 'prev_group->op_group' 매핑. 첫 op은 prev가 없어 제외된다."""
    ops = data['operations'].sort_values(['job_id', 'op_seq']).copy()
    ops['prev_group'] = ops.groupby('job_id')['op_group'].shift(1)
    ops = ops.dropna(subset=['prev_group'])
    ops['transition'] = ops['prev_group'] + '->' + ops['op_group']
    return ops.set_index('op_id')['transition'].to_dict()


class SAAEvaluator:
    """COMPOSITE rule 파라미터에 대한 SAA 평가기."""

    PARAM_NAMES = ('alpha', 'beta', 'gamma', 'delta')

    # transition 위험 테이블 산출에 쓰는 기존 rule 집합.
    RISK_PROBE_RULES = ('FIFO', 'SPT', 'LPT', 'MIN_QTIME', 'SPTSSU')

    def __init__(self, data,
                 reference_w=(1.0, 1.0, 1.0, 1.0),
                 n_ref=20,
                 obj_weights=(1.0, 1.0),
                 pm_hazard_threshold=0.1,
                 base_seed=0,
                 down_active=False,
                 pm_active=False):
        """
        Args:
            data: DataLoader.load_all_data() 결과
            reference_w: 정규화 기준이 되는 COMPOSITE 파라미터.
                         기본 (1,1,1,1) = COMPOSITE의 중립 출발점.
            n_ref: mk_ref, qv_ref 산출용 replication 수
            obj_weights: (w1, w2) — Makespan 항과 QV 항의 명시적 가중치.
                         기본 (1.0, 1.0) = 사용자가 의도한 1:1.
            pm_hazard_threshold: Scheduler에 넘길 PM 임계값 (PM 비활성 시 무영향)
            base_seed: seed_list 생성 기준값. seed_list[i] = base_seed + i
        """
        self.data = data
        self.pm_hazard_threshold = pm_hazard_threshold
        self.base_seed = base_seed
        self.w1, self.w2 = obj_weights
        self.reference_w = tuple(reference_w)
        self.n_ref = n_ref
        self._op2trans = _build_op_transition_map(data)

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
        #    reference_w로 COMPOSITE를 돌리므로 위 테이블이 먼저 주입돼 있어야 한다.
        self.mk_ref, self.qv_ref = self._compute_reference()

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
        RISK_PROBE_RULES를 각각 1회 실행해 transition별 qtime violation 시간을
        합산하고, 최댓값으로 나눠 [0,1] 위험 가중치 테이블을 만든다.

        위반이 한 건도 없으면 빈 테이블을 반환한다(c_transition 항이 무영향).
        """
        agg = {}
        for rule in self.RISK_PROBE_RULES:
            log_df = self._run_once(rule, self.base_seed)
            qv = log_df[log_df['event'] == 'qtime_over'].copy()
            if len(qv) == 0:
                continue
            qv['dur'] = qv['finish'] - qv['start']
            qv['transition'] = qv['op_id'].map(self._op2trans)
            for trans, dur in qv.groupby('transition')['dur'].sum().items():
                agg[trans] = agg.get(trans, 0.0) + float(dur)
        if not agg:
            return {}
        peak = max(agg.values())
        if peak <= 0:
            return {}
        return {t: v / peak for t, v in agg.items()}

    # ---- 목적함수 정규화 기준 ----
    def _compute_reference(self):
        """
        reference_w로 COMPOSITE를 n_ref회 실행해 mk_ref, qv_ref(평균)를 구한다.
        qv_ref가 0이면(위반 전무) 1.0으로 대체해 0 나눗셈을 막는다.
        """
        self._apply_params(self.reference_w)
        seeds = self._seed_list(self.n_ref)
        mks, qvs = [], []
        for s in seeds:
            log_df = self._run_once('COMPOSITE', s)
            mks.append(_makespan(log_df))
            qvs.append(_qtime_violation(log_df))
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
            'reference_w': self.reference_w,
            'mk_ref': self.mk_ref,
            'qv_ref': self.qv_ref,
            'obj_weights': (self.w1, self.w2),
            'transition_risk': dict(self.transition_risk),
        }
