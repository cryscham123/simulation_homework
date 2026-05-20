# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```powershell
pip install -r requirements.txt
cp .env_sample .env   # 이후 .env 값 채우기
```

## Running

단일 시뮬레이션 실행은 `result.ipynb`, 룰 비교 실험은 `result_compare.ipynb`를 Jupyter에서 실행한다. 장시간 실행(N_RUNS=50 이상)은 VS Code Jupyter 소켓 타임아웃으로 끊길 수 있으므로 JupyterLab을 사용한다:

```powershell
jupyter lab
```

## Environment Variables (`.env`)

| 변수 | 값 | 설명 |
|------|-----|------|
| `BASE_DATA_PATH` | `data` | CSV 데이터 디렉토리 경로 |
| `JOB_RULE` | `random` \| `FIFO` \| `SPT` \| `LPT` \| `MIN_QTIME` \| `SPTSSU` | Stocker dispatch 우선순위 규칙 |
| `PM_RULE` | `THRESHOLD` | 예방보전 규칙 |
| `PM_ACTIVE` | `True` \| `False` | 예방보전 활성화 여부 |
| `DOWN_ACTIVE` | `True` \| `False` | 고장 시뮬레이션 활성화 여부 |
| `PM_HAZARD_THRESHOLD` | `0.0~1.0` | Weibull 누적고장확률 PM 발동 임계값 |
| `TIME_UNIT` | `M` \| `H` \| `D` | 출력 시간 단위 (내부 계산은 항상 분) |

> `MACHINE_RULE`은 `.env_sample`에 남아있지만 현재 아키텍처에서 사용하지 않는다. 모든 dispatch는 `JOB_RULE`로만 제어된다.

## Architecture

### 시뮬레이션 흐름

```
Scheduler.__init__
  ├── Machine × 60개 생성 (down/PM 프로세스 시작)
  ├── Stocker 생성 (machine_signal Store 구독)
  ├── 모든 Machine을 machine_signal에 put → t=0에 Stocker.__waiting_machines 초기화
  ├── Job × 100개 생성 → job.release() 프로세스 시작
  └── __chk_job_waiting() 루프 시작

Job.release()
  → release_time 대기 후 job_events.put(self)

Scheduler.__chk_job_waiting()
  → job_events에서 job 수신
  → RELEASED: __matching_machine(job) 호출
  → COMPLETED: terminated_jobs++

Scheduler.__matching_machine(job)
  → op 완료 직후이면 job.start_qtime_chk() 시작 (첫 release는 qtime 미적용)
  → stocker.add_job(job) 프로세스 시작
```

### Stocker 디스패치 로직 (듀얼 큐 랑데부)

두 경로가 항상 상호 배타적으로 동작한다:

```
add_job(job):
  waiting_machines에 같은 group machine이 있으면
    → 즉시 dispatch (FilterStore 경유 X)
  없으면
    → FilterStore.put(job) 대기

wait_until_machine_ready():  ← machine 완료 시 machine_signal 수신
  FilterStore에 같은 group job이 있으면
    → JOB_RULE로 선택 후 FilterStore.get(job) → dispatch
  없으면
    → waiting_machines에 machine 추가
```

### Q-Time 메커니즘

- `qtime`은 operation 단위로 설정되며 **직전 op 완료 시점부터 카운트** 시작
- `job.py`에서 `qtime <= 0`은 `inf`로 치환 → qtime 없는 op은 위반 불가
- Machine이 setup을 시작하는 순간 `machine.run()`이 `job.interrupt_qtime()` 호출 → 위반 타이머 중단
- **주요 위반 발생 transition**: `G3→G1` (600분), `G4→G1` (480분), `G1→G3/G4` (1440분)
- G1(30대)이 bottleneck이므로 G1 앞 대기가 길어질수록 violation 증가

### Machine 고장/PM

- Weibull 분포 `F(t) = 1 - exp(-(t/λ)^k)`로 고장 시간 샘플링
- PM은 `F(t) = PM_HAZARD_THRESHOLD`가 되는 시점에 발동
- 고장이 PM보다 먼저 발생하면 PM 프로세스를 interrupt → `repair()` 실행
- PM 성공 시 down 프로세스 재시작 (Weibull 리셋)
- `simpy.PreemptiveResource`로 repair가 working을 선점

### 데이터 구조

| 파일 | 핵심 컬럼 |
|------|----------|
| `jobs.csv` | `job_id`, `job_type`(P1/P2), `release_time`, `due_date` |
| `operations.csv` | `job_id`, `op_id`, `op_seq`, `op_group`(G1~G5), `qtime`(분) |
| `machines.csv` | `machine_id`, `machine_group`(G1~G5) |
| `machine_failure.csv` | `machine_id`, `shape parameter`(k), `scale parameter`(λ, 분 단위), `repair_time`, `pm_duration` |
| `setup_times.csv` | `machine_group`, `from_job_type`, `to_job_type`, `setup_time` |
| `operation_machine_map.csv` | `op_id`, `machine_id`, `process_time` |

Machine group 구성: G1(30대) · G2(4대) · G3(16대) · G4(8대) · G5(2대)

### EventLogger

내부 시간은 항상 **분(minute)** 단위. `TIME_UNIT` 환경변수에 따라 `logs` property 반환 시 변환.  
`log_event_start()` → index 반환 → `log_event_finish(index)`로 종료 시각 기록.  
`index = -1`이면 `log_event_finish`는 no-op.

### result_compare.ipynb 구조

1. **환경 설정** → 2. **모듈 import** → 3. **데이터 로드** → 4. **`run_simulation(job_rule, pm_rule)` 함수 정의**  
5. **시뮬레이션 루프** (`JOB_RULES × PM_RULES × N_RUNS`): `metrics_df` 생성  
   - `transition_violation`: `(job_type, transition)` 별 qtime 위반 시간 합계 (Series)  
6. **시각화 셀들** (각 셀 독립 실행 가능): Makespan / Qtime Violation / PM&Repair / Utilization / Flowtime&Tardiness
