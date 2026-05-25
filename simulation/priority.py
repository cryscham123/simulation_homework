"""
COMPOSITE dispatching rule의 priority score 계산 모듈 (Index Policy).

수리적 근거:
  Index Policy(복합 priority score)를 선형결합으로 정의한다.

  π(j) = α·norm(α_term(j))
       + β·norm(slack_qtime(j))
       + γ·norm(c_transition(j))
       + δ·norm(setup(j))

  - α_term은 환경변수 COMPOSITE_ALPHA_TERM으로 선택한다:
      * 'WAIT_TIME' (기본): WAITING 상태 누적 대기 시간(분). 클수록 우선
        (FIFO 일관성)이 되도록 음수로 반환한다 → norm 후 가장 오래 기다린
        job이 가장 작은 값 → 우선. 같은 op_group candidate라도 stocker
        도착 시점이 달라 자연스럽게 variance가 생긴다.
      * 'CRITICAL_RATIO': remain_qtime / next_PT. 작을수록 위험 → 우선.
        slack을 next PT로 정규화한 형태. qtime=inf는 None → 모집단 제외.
    α 자리에 어떤 정보 항이 들어갈지를 단계별 가설 검증으로 갈아끼우기
    위한 hook이다(grid 차원·환경변수 이름은 유지 → 노트북 호환).

  - 각 항은 candidate 집합 내 min-max 정규화 → 가중치 α,β,γ,δ는
    스케일 보정이 아니라 순수 '중요도'만 담당한다.
  - π는 'min 선택 = 낮을수록 우선'. setup은 작을수록 우선(SPT 일관성),
    slack은 작을수록 위험하므로 그대로 작은 값 → 우선,
    c_transition은 '다음 transition의 위험도'를 부정 부호로 넣어
    위험한 transition으로 들어가는 job이 작은 값 → 우선이 되게 한다.

  주의: 메커니즘 2(WIP Cap)는 데이터 구조상 G3/G4 op의 다음 공정이
  100% G1이어서 'candidate 중 일부만 후순위화'가 불가능하므로 제외했다.

파라미터는 환경변수로 노출되어 SAA 튜닝에서 sweep 가능하다.
"""
import os

EPS = 1e-9

# remain_qtime을 유한 범위로 묶기 위한 클리핑 상한.
SLACK_CLIP = 1e7

# transition 위험 가중치 테이블. saa_evaluator가 baseline 측정 후 주입한다.
# key: 'prevGroup->curGroup', value: [0,1] 위험도. 비어 있으면 c_transition=0.
_TRANSITION_RISK = {}


def set_transition_risk_table(table):
    """saa_evaluator에서 측정한 transition 위험 테이블을 모듈에 주입한다."""
    global _TRANSITION_RISK
    _TRANSITION_RISK = dict(table) if table else {}


def get_transition_risk_table():
    """현재 주입된 transition 위험 테이블의 사본."""
    return dict(_TRANSITION_RISK)


def _get_params():
    """환경변수에서 COMPOSITE rule 파라미터를 읽는다."""
    return {
        'alpha': float(os.getenv('COMPOSITE_ALPHA', '1.0')),
        'beta': float(os.getenv('COMPOSITE_BETA', '1.0')),
        'gamma': float(os.getenv('COMPOSITE_GAMMA', '1.0')),
        'delta': float(os.getenv('COMPOSITE_DELTA', '1.0')),
    }


def _min_max_norm(values):
    """
    candidate 집합 내 min-max 정규화 → [0, 1].
    candidate가 1개이거나 전부 동일하면 모두 0 (해당 항이 우선순위에 무영향).
    """
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= EPS:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def _min_max_norm_skip_none(values):
    """
    None을 '위험 없음(=가장 안전, 후순위)'으로 취급하는 inf-aware min-max 정규화.

    slack 항에서 qtime 없는 op(remain_qtime=inf)을 정규화 모집단에 넣으면,
    span이 거대해져 유한 위험값들이 전부 0 근처로 뭉개진다(위험 변별 소실).
    → None은 모집단에서 제외하고 min-max를 잡은 뒤 norm=1.0(가장 안전)을
    부여한다. slack 항은 'min 선택 = 낮을수록 우선'이므로 norm=1.0이면
    후순위가 된다. 이로써 '위험한 job들끼리의 상대 위험도'가 정규화 후에도
    보존되고, 위반 불가 op은 위험 job보다 항상 뒤로 밀린다.

    단, 유한 위험 job이 하나도 없으면(모두 inf) 이 항은 무의미하므로 0 처리.
    """
    finite = [v for v in values if v is not None]
    if not finite:
        return [0.0] * len(values)
    lo, hi = min(finite), max(finite)
    span = hi - lo
    out = []
    for v in values:
        if v is None:
            out.append(1.0)
        elif span <= EPS:
            out.append(0.0)
        else:
            out.append((v - lo) / span)
    return out


def _slack_raw(job):
    """
    slack_qtime 그 자체(= remain_qtime). 작을수록 위험.

    π는 'min 선택 = 낮을수록 우선'이므로, 위험한 job(작은 slack)이 작은
    항 값을 받아야 우선순위가 올라간다. 따라서 +slack을 그대로 쓴다
    (정규화 후 위험 job → norm≈0 → score 낮음 → 우선).

    remain_qtime이 inf(qtime 없는 op → 위반 불가)이면 None을 반환해
    정규화 모집단에서 제외하고, 가장 안전하도록 norm=1.0을 부여한다.
    """
    remain = job.get_remain_qtime()
    if remain == float('inf'):
        return None
    return max(min(remain, SLACK_CLIP), -SLACK_CLIP)


def _critical_ratio(job, machine):
    """
    Critical Ratio (A-3): remain_qtime / next_process_time. 작을수록 우선.

    근거: slack(remain_qtime)은 시간의 절대 여유만 보지만, CR은 그 여유를
    '다음 작업이 얼마나 시간을 잡아먹는지'로 정규화한다. 같은 잔여 qtime이라도
    next PT가 긴 job이 더 위험(시간이 빠듯).

    부호: π는 'min 선택 = 우선'. 작은 CR(시간 압박↑) 그대로 작은 값 → 우선.

    qtime이 inf(=qtime 없는 op, 위반 불가)이면 None을 반환해 정규화 모집단에서
    제외하고 norm=1.0(최후순위) 부여(slack과 동일 처리).

    next_process_time은 현재 머신 기준 PT. PT가 0에 가까우면 division 보호.
    """
    remain = job.get_remain_qtime()
    if remain == float('inf'):
        return None
    pt = machine.get_process_time(job.get_current_operation())
    return float(remain) / max(float(pt), EPS)


def _waiting_time_neg(job):
    """
    WAITING 상태 누적 대기 시간(분)을 음수로 반환 (FIFO 일관성, A-2).

    근거: result_compare에서 FIFO가 makespan 최저(213.5h)이고 QV도 낮음.
    "오래 기다린 job을 먼저 dispatch" 신호가 큐 평탄화 → G1 throughput 안정화
    → G1→G4 위반(현재 미해결 표적) 감소에 기여한다는 가설.

    부호: π는 'min 선택 = 우선'이므로, 가장 오래 기다린 job이 가장 작은 값을
    받아야 우선이 된다 → -waiting_time. min-max norm 후에도 부호가 유지되어
    오래 기다린 candidate가 norm≈0 → score 낮음 → 우선.

    candidate 변별력: 같은 op_group candidate라도 stocker 도착 시점이 다르면
    waiting_time이 다르므로 norm 후 0~1로 자연스럽게 펼쳐진다.
    """
    return -float(job.get_waiting_time())


def _transition_risk(job):
    """
    job이 '현재 op를 끝낸 뒤 들어갈 다음 transition'의 위험도.

    부호 규약: π는 'min 선택 = 우선'이므로, 위험한 transition으로 들어가는
    job이 작은 항 값을 받아야 우선이 된다 → 위험도를 음수로 반환해
    위험할수록 score를 끌어내린다(정규화 전).

    다음 op이 없으면(현재 op이 마지막) 위험 없음 → 0.0.
    테이블에 없는 transition도 0.0.
    """
    trans = job.get_next_transition()
    if trans is None:
        return 0.0
    return -float(_TRANSITION_RISK.get(trans, 0.0))


def _alpha_term_values(candidates, machine):
    """
    α 항의 raw 값 리스트. COMPOSITE_ALPHA_TERM 환경변수로 정의를 전환한다.

      'WAIT_TIME'(기본): -job.get_waiting_time() — FIFO 일관성
      'CRITICAL_RATIO' : remain_qtime / next_PT — CR.
                         qtime=inf인 job은 None → 정규화 모집단 제외, norm=1.0
    """
    term = os.getenv('COMPOSITE_ALPHA_TERM', 'WAIT_TIME').upper()
    if term == 'CRITICAL_RATIO':
        return [_critical_ratio(j, machine) for j in candidates]
    # default: WAIT_TIME
    return [_waiting_time_neg(j) for j in candidates]


def _alpha_term_uses_skip_none():
    """현재 α-term이 None을 모집단에서 제외해야 하는지 (slack과 동일 처리)."""
    return os.getenv('COMPOSITE_ALPHA_TERM', 'WAIT_TIME').upper() == 'CRITICAL_RATIO'


def composite_select(candidates, machine):
    """
    COMPOSITE rule: candidate 중 priority score π가 최소인 job을 선택.

    Args:
        candidates: dispatch 후보 job 리스트 (모두 machine.group과 동일 op group)
        machine: dispatch 대상 machine

    Returns:
        선택된 Job
    """
    p = _get_params()

    a_raw = _alpha_term_values(candidates, machine)
    slack = [_slack_raw(j) for j in candidates]
    trans = [_transition_risk(j) for j in candidates]
    setup = [machine.get_setup_time(j.job_type) for j in candidates]

    # α-term이 CRITICAL_RATIO이면 None(qtime=inf)이 섞이므로 skip_none 정규화 사용
    n_a = (_min_max_norm_skip_none(a_raw)
           if _alpha_term_uses_skip_none()
           else _min_max_norm(a_raw))
    n_slack = _min_max_norm_skip_none(slack)  # inf(위반 불가) op은 모집단 제외
    n_trans = _min_max_norm(trans)
    n_setup = _min_max_norm(setup)

    best_idx, best_score = 0, float('inf')
    for i in range(len(candidates)):
        score = (
            p['alpha'] * n_a[i]
            + p['beta'] * n_slack[i]
            + p['gamma'] * n_trans[i]
            + p['delta'] * n_setup[i]
        )
        if score < best_score:
            best_idx, best_score = i, score

    return candidates[best_idx]
