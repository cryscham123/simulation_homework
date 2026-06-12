# SimPy 기반 반도체 공정 스케줄링 시뮬레이터

simPy를 사용해 반도체 공정의 작업 흐름, 설비 고장, 예방 보전(PM), Q-Time 위반을 시뮬레이션하는 프로젝트.

## 주요 기능

- CSV 기반 설비, 작업, 공정, 셋업 시간, 고장 파라미터 로딩
    - 데이터: https://drive.google.com/drive/u/0/folders/1BLy4dLHxWocqR834kZDDHDh7nR_VcyyB
- Dispatching: `FIFO`, `SPT`, `LPT`, `MIN_QTIME`, `SPTSSU` 룰 지원
- GA 기반 job sequence, operation-machine 할당, PM threshold 최적화
    - GA 시뮬레이션 working provision: https://github.com/cryscham123/simulation_external_excute
- 이벤트 로그 기반 makespan, Q-Time violation 계산 및 Gantt chart, boxplot 시각화

## 설치

```bash
pip install -r requirements.txt
```

- 의존성 패키지는 `requirements.txt`에 명시되어 있습니다.

```bash
cp .env_sample .env
```

- 환경 변수 파일은 샘플을 복사해 사용합니다.
- 주요 환경 변수는 다음과 같습니다.


| 변수 | 설명 | 예시 |
| --- | --- | --- |
| `BASE_DATA_PATH` | CSV 데이터 디렉터리 | `data/large_data` |
| `PM_ACTIVE` | 예방 보전 활성화 여부 | `True` / `False` |
| `DOWN_ACTIVE` | 설비 고장 활성화 여부 | `True` / `False` |
| `TIME_UNIT` | 출력 시간 단위 | `M`, `H`, `D` |
| `PM_HAZARD_THRESHOLD` | 룰 기반 PM hazard threshold | `0.1` |
| `JOB_RULE` | 룰 기반 job Dispatching 규칙 | `MIN_QTIME` |
| `PM_LEVELS` | GA에서 탐색할 PM threshold 후보 | `[0.01, 0.03, 0.05]` |
| `POP_SIZE` | GA population 크기 | `100` |
| `N_GENERATIONS` | GA 세대 수 | `100` |
| `ALPHA` | `makespan + alpha * qtime_violation` 가중치 | `1.0` |

- 아래는 예시 환경 변수 입니다.

```text
PM_ACTIVE=True
DOWN_ACTIVE=True
MACHINE_RULE=INDEX
BASE_DATA_PATH=data/large_data
TIME_UNIT=H

# 룰베이스
PM_HAZARD_THRESHOLD=0.1
JOB_RULE=MIN_QTIME
PM_RULE=THRESHOLD

# GA
PM_LEVELS=[0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]
POP_SIZE=100
N_GENERATIONS=1000
CROSSOVER_RATE=0.8
MUT_JOB=0.01
MUT_MACHINE=0.0005
MUT_PM=0.005
TOURNAMENT_K=3
N_ELITES=1
ALPHA=1
SEED=993322

STAGNATION_PATIENCE=400
MUTATION_BOOST=30.0
IMMIGRANT_RATIO=0.5
BOOST_DURATION=5
BOOST_COOLDOWN=20
```

- 데이터는 위 명시된 google cloud에서 다운로드 가능합니다.
- 저장 경로는 사용자 편의에 맞게 저장하면 되지만, .env 파일의 `BASE_DATA_PATH`와 일치해야 합니다.
- 입력 데이터는 아래와 같습니다.
    1. `machines.csv`
    2. `jobs.csv`
    3. `machine_failure.csv`
    4. `operation_machine_map.csv`
    5. `operations.csv`
    6. `setup_times.csv`

## 실행

현재 프로젝트는 노트북 중심으로 실행합니다.

- `result.ipynb`: 단일 룰 기반 시뮬레이션 실행
- `result_compare.ipynb`: 여러 룰 기반 설정 비교
- `result_ga.ipynb`: GA 기반 최적화 실행

## 프로젝트 구조

```text
simpy/
├── algorithms/
│   └── genetic/              # GA 인코딩, 디코딩, 평가, 연산자, 실행 로직
├── reports/                  # 최종 보고서
├── simulation/
│   ├── job.py                # Job 상태 및 operation 흐름
│   ├── machine.py            # Machine 상태, 고장, PM, 작업 처리
│   ├── scheduler.py          # Job-machine 매칭 및 이벤트 흐름 제어
│   └── stocker.py            # 대기 job dispatching
├── utils/
│   ├── data_loader.py        # CSV 데이터 로딩
│   ├── event_logger.py       # 이벤트 로그 기록
│   └── visualizer.py         # Gantt chart 생성
├── result.ipynb              # 룰 기반 단일 실행
├── result_compare.ipynb      # 룰 기반 비교 실험
├── result_ga.ipynb           # GA 실험
├── requirements.txt
└── README.md
```

