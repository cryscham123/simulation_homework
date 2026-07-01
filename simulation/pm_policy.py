import os
import json
import math
from functools import lru_cache
from math import inf, gamma

import scipy.integrate as _spi
import scipy.optimize as _spo


# 내부 참조 브랜치(simulation_teamwork/simulation_teamwork)의 MTTF rule 기본값.
# env (MTTF_RISK_GROUPS / MTTF_RISK_COEFFICIENTS)로 override 가능.
_DEFAULT_MTTF_RISK_GROUPS = {
    "G1": "medium", "G2": "low", "G3": "high", "G4": "high", "G5": "medium"
}
_DEFAULT_MTTF_RISK_COEF = {"high": 0.3, "medium": 0.5, "low": 0.7}


def _threshold_pm_time(shape, scale, threshold):
    """F(t) = threshold 시점. t = λ · (-ln(1-threshold))^(1/k)."""
    if shape <= 0 or scale <= 0:
        return inf
    if threshold <= 0:
        return 0.0
    if threshold >= 1:
        return inf
    return scale * ((-math.log(1.0 - threshold)) ** (1.0 / shape))


def _weibull_survival(t, shape, scale):
    """R(t) = exp(-(t/λ)^k)."""
    if t <= 0:
        return 1.0
    return math.exp(-((t / scale) ** shape))


@lru_cache(maxsize=128)
def _age_replacement_optimal(shape, scale, repair_time, pm_duration):
    """
    Barlow–Proschan age-replacement T* = argmin g(T).

    g(T) = [pm_duration·R(T) + repair_time·F(T)] / ∫₀ᵀ R(u) du

    k ≤ 1이면 PM 무의미 → inf.
    repair_time ≤ pm_duration이면 고장 수리가 PM보다 싸므로 PM 무의미 → inf.
    """
    if shape <= 1.0 or scale <= 0:
        return inf
    if repair_time <= pm_duration:
        return inf

    mttf = scale * gamma(1.0 + 1.0 / shape)

    def g(T):
        if T <= 0:
            return inf
        R = _weibull_survival(T, shape, scale)
        F = 1.0 - R
        integral, _ = _spi.quad(_weibull_survival, 0.0, T,
                                args=(shape, scale), limit=100)
        if integral <= 1e-12:
            return inf
        return (pm_duration * R + repair_time * F) / integral

    # MTTF의 0.05~5배 범위에서 탐색. 충분히 넓은 unimodal 영역.
    res = _spo.minimize_scalar(
        g,
        bounds=(0.05 * mttf, 5.0 * mttf),
        method='bounded',
        options={'xatol': max(1.0, mttf * 1e-4)},
    )
    return float(res.x) if res.success else inf


def _mttf_pm_time(shape, scale, group):
    """
    MTTF:
        pm_interval = pm_coefficient × MTTF
        MTTF        = scale · Γ(1 + 1/shape)
        pm_coefficient = risk_coefficients[ group_risk_assignment[group] ]

    group_risk_assignment과 risk_coefficients는 env (MTTF_RISK_GROUPS,
    MTTF_RISK_COEFFICIENTS) JSON으로 override 가능. 미지정 시 참조 브랜치의
    default를 그대로 사용한다.

    group이 risk 매핑에 없거나 risk_type이 coefficient 매핑에 없으면 inf
    (= PM 미발동) — 정의되지 않은 그룹에 임의의 PM 주기를 부여하지 않는다.
    """
    if shape <= 0 or scale <= 0:
        return inf

    raw_groups = os.getenv('MTTF_RISK_GROUPS')
    raw_coef = os.getenv('MTTF_RISK_COEFFICIENTS')
    group_risk = json.loads(raw_groups) if raw_groups else _DEFAULT_MTTF_RISK_GROUPS
    risk_coef = json.loads(raw_coef) if raw_coef else _DEFAULT_MTTF_RISK_COEF

    risk_type = group_risk.get(group)
    if risk_type is None:
        return inf
    coef = risk_coef.get(risk_type)
    if coef is None:
        return inf

    mttf = scale * gamma(1.0 + 1.0 / shape)
    return float(coef) * mttf


def calculate_pm_time(rule, shape, scale, repair_time, pm_duration, threshold,
                      group=None):
    """
    PM rule에 따라 PM 발동 시각을 반환한다.

    Args:
        rule: 'THRESHOLD' | 'AGE_REPLACEMENT' | 'MTTF'
        shape, scale: Weibull 파라미터
        repair_time, pm_duration: machine_failure data에서 온 cost
        threshold: PM_HAZARD_THRESHOLD (THRESHOLD rule 전용)
        group: machine_group (MTTF rule 전용 — risk 매핑 lookup용)
    """
    rule = (rule or 'THRESHOLD').upper()
    if rule == 'THRESHOLD':
        return _threshold_pm_time(shape, scale, threshold)
    if rule == 'AGE_REPLACEMENT':
        return _age_replacement_optimal(shape, scale, repair_time, pm_duration)
    if rule == 'MTTF':
        return _mttf_pm_time(shape, scale, group)
    raise ValueError(f"Unknown PM_RULE: {rule}. Must be THRESHOLD, AGE_REPLACEMENT, or MTTF.")
