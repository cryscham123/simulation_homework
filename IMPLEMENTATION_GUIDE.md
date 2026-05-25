# PM Rule (MTTF) 구현 가이드

## 📋 개요

시뮬레이션 시스템에 **MTTF 기반 PM 규칙**을 추가했습니다. 기존의 THRESHOLD 규칙과 함께 사용할 수 있으며, 두 규칙을 자유롭게 비교 분석할 수 있습니다.

---

## 🔧 구현 내용

### 1. **source/pm_policy.py** - MTTF 규칙 함수 추가

#### 새로운 함수들:
- **`apply_mttf_pm_rule()`**: MTTF 기반 PM 규칙 적용
  - Machine group별 위험도(high/medium/low) 지정
  - PM Interval = 계수 × MTTF 계산
  - 반환: (pm_interval_dict, machine_policy_df, group_policy_df)

- **`get_pm_interval_by_rule()`**: 통합 PM 규칙 인터페이스
  - PM_RULE 파라미터로 THRESHOLD 또는 MTTF 선택
  - 자동으로 적절한 계산 로직 수행
  - 반환: (pm_interval_dict, policy_df1, policy_df2)

### 2. **.env_sample 및 .env** - 환경 설정 확장

#### 새로운 파라미터:
```ini
PM_RULE=(THRESHOLD/MTTF)

# MTTF 규칙 파라미터 (PM_RULE=MTTF일 경우)
MTTF_RISK_GROUPS={"G1": "medium", "G2": "low", "G3": "high"}
MTTF_RISK_COEFFICIENTS={"high": 0.3, "medium": 0.5, "low": 0.7}
```

### 3. **result.ipynb** - PM 규칙 선택 기능 추가

#### 변경 사항 (최소):
- Block 1: MTTF 파라미터 로드
- Block 3: 새 함수들 import
- Block 7: PM 규칙에 따른 pm_interval 설정

**사용법:**
```python
# .env에서 자동 로드됨
PM_RULE = "MTTF"  # 또는 "THRESHOLD"

# 나머지는 자동으로 처리됨
```

### 4. **result_compare.ipynb** - PM 규칙 선택 기능 추가

기존 구조 유지, 최소 수정:
- Block 1: MTTF 파라미터 추가
- Block 3: 함수 import 확장

### 5. **source/pm_compare.ipynb** - 완전히 새로운 분석 도구

#### 11개 블록 구조:
1. **Block 1**: 환경설정 (PM_RULES_TO_COMPARE 설정)
2. **Block 2**: 모듈 Import
3. **Block 3**: 데이터 로드
4. **Block 4**: Machine Group별 특성 요약
5. **Block 5**: Weibull 분포 및 MTTF 분석
6. **Block 6**: PM Interval 설정 (THRESHOLD vs MTTF)
7. **Block 7**: 시뮬레이션 실행 (PM 규칙별 비교)
8. **Block 8**: 결과 집계 및 통계
9. **Block 9**: Makespan 분포 시각화
10. **Block 10**: Q-time Violation 분포 시각화
11. **Block 11**: PM/Repair 메트릭 시각화
12. **Block 12**: 통합 성과 지표 시각화

---

## 💡 사용 방법

### **시나리오 1: 특정 PM 규칙으로 시뮬레이션 실행 (result.ipynb)**

```python
# .env 파일 설정
PM_RULE=MTTF
MTTF_RISK_GROUPS={"G1": "high", "G2": "medium", "G3": "low"}
MTTF_RISK_COEFFICIENTS={"high": 0.3, "medium": 0.5, "low": 0.7}

# result.ipynb 실행
# 자동으로 MTTF 규칙이 적용되어 시뮬레이션 수행
```

### **시나리오 2: 두 PM 규칙 비교 분석 (pm_compare.ipynb)**

```python
# Block 1에서 설정
PM_RULES_TO_COMPARE = ["THRESHOLD", "MTTF"]
JOB_RULES = ["MIN_QTIME", "FIFO", "SPT"]
N_REPLICATIONS = 10

# 모든 블록 순차 실행
# 결과: PM 규칙별, 디스패칭 규칙별 성과 지표 비교
# - Makespan 분포
# - Q-time violation
# - PM/Repair 통계
# - 통합 성과 지표
```

### **시나리오 3: 계수 조정 및 재실행**

```python
# .env 파일에서 계수 수정
MTTF_RISK_COEFFICIENTS={"high": 0.25, "medium": 0.45, "low": 0.65}

# pm_compare.ipynb 재실행
# 다양한 계수 조합 효과 비교 가능
```

---

## 📊 예상 결과

### result.ipynb 실행 시:
- 선택한 PM 규칙으로 시뮬레이션 수행
- MTTF 규칙 선택 시: Group별 PM Policy 표 출력
- 기존과 동일한 KPI 및 통계 결과

### pm_compare.ipynb 실행 시:
- **표**: Machine Group별 특성 + MTTF 값
- **표**: PM 규칙별 Makespan/Q-time/PM 통계
- **그래프 1**: Makespan 분포 (PM 규칙별, 디스패칭 규칙별)
- **그래프 2**: Q-time Violation 분포
- **그래프 3**: PM/Repair 메트릭 (4개 그래프)
- **그래프 4**: 통합 성과 지표 (Total Score)
- **결론**: 최적 규칙 조합 추천

---

## 🔍 주요 특징

✅ **최소 수정**: 기존 원본 파일 거의 수정 안 함  
✅ **.env 연동**: 모든 설정을 .env에서 관리  
✅ **모듈화**: PM 규칙을 별도 파일에서 중앙 관리  
✅ **확장성**: 향후 자동 계수 산정 가능하도록 설계  
✅ **비교 분석**: 두 규칙을 정량적으로 비교 가능  

---

## 📝 추가 정보

### MTTF 계산 원리
```
MTTF = scale × Gamma(1 + 1/shape)
PM_Interval = MTTF × coefficient
```

### PM 규칙 선택 흐름
```
get_pm_interval_by_rule()
  ├─ PM_RULE = "THRESHOLD"
  │   └─ weibull_time_at_failure_prob()로 계산
  └─ PM_RULE = "MTTF"
      └─ apply_mttf_pm_rule()로 계산
```

### 디렉토리 구조
```
simulation_teamwork/
├── source/
│   ├── pm_policy.py          (확장됨)
│   ├── pm_compare.ipynb       (신규)
│   └── ...
├── result.ipynb              (최소 수정)
├── result_compare.ipynb       (최소 수정)
├── .env                       (확장됨)
├── .env_sample               (신규)
└── ...
```

---

## 🚀 다음 단계

향후 개선 사항:
- [ ] Machine 특성(reliability, substitution ratio)을 고려한 자동 계수 산정
- [ ] Q-time 민감도를 반영한 계수 조정
- [ ] 다양한 PM 규칙 추가 (Age-Based, Condition-Based 등)
- [ ] ML 기반 최적 계수 탐색

