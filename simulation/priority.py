import os

EPS = 1e-9

# remain_qtime을 유한 범위로 묶기 위한 클리핑 상한.
SLACK_CLIP = 1e7

# transition 위험 가중치 테이블. saa_evaluator가 데이터 기반으로 산출 후 주입한다.
_TRANSITION_RISK = {}

# normalizer 종류 식별자
NORM_MINMAX = 'minmax'
NORM_MINMAX_SKIP_NONE = 'minmax_skip_none'


def set_transition_risk_table(table):
    """saa_evaluator에서 산출한 transition 위험 테이블을 모듈에 주입한다."""
    global _TRANSITION_RISK
    _TRANSITION_RISK = dict(table) if table else {}


def get_transition_risk_table():
    """현재 주입된 transition 위험 테이블의 사본."""
    return dict(_TRANSITION_RISK)


# ---- weight / slot 환경변수 ----
SLOT_NAMES = ('alpha', 'beta', 'gamma', 'delta')

# 슬롯별 기본 항 (env var 미설정 시 기존 동작과 동일)
_DEFAULT_SLOT_TERMS = {
    'alpha': 'PT',
    'beta':  'SLACK',
    'gamma': 'C_TRANSITION',
    'delta': 'SETUP',
}


def _get_weights():
    """환경변수에서 슬롯별 가중치를 읽는다."""
    return {s: float(os.getenv(f'COMPOSITE_{s.upper()}', '1.0'))
            for s in SLOT_NAMES}


def _get_slot_terms():
    """환경변수에서 슬롯 → 항 이름 매핑을 읽는다."""
    return {s: os.getenv(f'COMPOSITE_{s.upper()}_TERM',
                          _DEFAULT_SLOT_TERMS[s]).upper()
            for s in SLOT_NAMES}


# ---- normalizer ----
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
    부여한다. 'min 선택 = 낮을수록 우선'이므로 norm=1.0이면 후순위가 된다.

    단, 유한 위험 job이 하나도 없으면(모두 None) 이 항은 무의미하므로 0 처리.
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


_NORMALIZERS = {
    NORM_MINMAX: _min_max_norm,
    NORM_MINMAX_SKIP_NONE: _min_max_norm_skip_none,
}


# ---- 항(term) 정의 ----
# 모든 raw_fn은 (candidates, machine) -> List[float | None]
# 부호 규약: "작은 값일수록 우선" (min 선택과 일치) 이 되도록 raw에서 부호를 맞춘다.
#
# 새 항 추가는 함수 정의 후 TERM_REGISTRY에 한 줄 등록만 하면 끝.

def _term_pt(candidates, machine):
    """PT: 현재 candidate op의 process time. 짧을수록 우선."""
    return [float(machine.get_process_time(j.get_current_operation()))
            for j in candidates]


def _term_slack(candidates, machine):
    """
    SLACK: remain_qtime. 작을수록(여유 없을수록) 우선.
    remain_qtime이 inf(qtime 없음 → 위반 불가)인 candidate는 None으로
    반환 → 정규화 모집단에서 제외하고 가장 후순위.
    """
    out = []
    for j in candidates:
        r = j.get_remain_qtime()
        if r == float('inf'):
            out.append(None)
        else:
            out.append(max(min(r, SLACK_CLIP), -SLACK_CLIP))
    return out


def _term_setup(candidates, machine):
    """SETUP: machine 기준 setup time. 짧을수록 우선."""
    return [float(machine.get_setup_time(j.job_type)) for j in candidates]


def _term_c_transition(candidates, machine):
    """
    C_TRANSITION: 다음 transition의 사전 위험도(데이터 기반 [0,1]).
    위험할수록 우선이 되도록 부호를 반전해 음수로 반환한다.
    다음 op 없거나 테이블 미등록 transition은 0.
    """
    out = []
    for j in candidates:
        trans = j.get_next_transition()
        if trans is None:
            out.append(0.0)
        else:
            out.append(-float(_TRANSITION_RISK.get(trans, 0.0)))
    return out


def _term_completion_fast(candidates, machine):
    """
    COMPLETION_FAST: 진행이 많이 된 job을 우선 (빠른 job 우선 투입).
    completion_ratio = cur_seq / total_ops. 높을수록 우선이 되도록 부호 반전.
    """
    out = []
    for j in candidates:
        total = j.total_ops
        if total <= 0:
            out.append(0.0)
        else:
            out.append(-(j.cur_seq / total))
    return out


def _term_completion_slow(candidates, machine):
    """
    COMPLETION_SLOW: 진행이 느린 job을 우선 (덜 끝난 job 보호).
    completion_ratio = cur_seq / total_ops. 낮을수록 우선 → 그대로.
    """
    out = []
    for j in candidates:
        total = j.total_ops
        if total <= 0:
            out.append(0.0)
        else:
            out.append(j.cur_seq / total)
    return out


def _term_waiting(candidates, machine):
    """
    WAITING: 현재 WAITING 상태에서의 누적 대기 시간. 오래 기다린 job 우선.
    "오래 기다린 = 우선"이 되도록 음수로 반환.
    """
    return [-float(j.get_waiting_time()) for j in candidates]


# (raw_fn, normalizer) 쌍 등록.
# 새 후보 항 추가는 여기에 한 줄 추가하면 끝.
TERM_REGISTRY = {
    'PT':              (_term_pt,              NORM_MINMAX),
    'SLACK':           (_term_slack,           NORM_MINMAX_SKIP_NONE),
    'SETUP':           (_term_setup,           NORM_MINMAX),
    'C_TRANSITION':    (_term_c_transition,    NORM_MINMAX),
    'COMPLETION_FAST': (_term_completion_fast, NORM_MINMAX),
    'COMPLETION_SLOW': (_term_completion_slow, NORM_MINMAX),
    'WAITING':         (_term_waiting,         NORM_MINMAX),
}


def available_terms():
    """등록된 항 이름 리스트."""
    return list(TERM_REGISTRY.keys())


def composite_select(candidates, machine):
    """
    COMPOSITE rule: candidate 중 priority score π가 최소인 job을 선택.

    슬롯(α/β/γ/δ)별로 어떤 항을 끼울지는 환경변수
      COMPOSITE_<SLOT>_TERM (값: TERM_REGISTRY key)
    로 결정한다. 미설정이면 _DEFAULT_SLOT_TERMS에 따른다.
    """
    weights = _get_weights()
    slots = _get_slot_terms()

    norm_by_slot = {}
    for slot, term_name in slots.items():
        if term_name not in TERM_REGISTRY:
            raise KeyError(
                f"COMPOSITE_{slot.upper()}_TERM='{term_name}' 은 등록된 항이 아니다. "
                f"사용 가능: {available_terms()}"
            )
        raw_fn, norm_kind = TERM_REGISTRY[term_name]
        raw = raw_fn(candidates, machine)
        norm_by_slot[slot] = _NORMALIZERS[norm_kind](raw)

    best_idx, best_score = 0, float('inf')
    for i in range(len(candidates)):
        score = sum(weights[s] * norm_by_slot[s][i] for s in SLOT_NAMES)
        if score < best_score:
            best_idx, best_score = i, score

    return candidates[best_idx]
