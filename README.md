## 프로젝트 구조

```
simpy/
├── data/                    # 시뮬레이션 데이터 파일들
├── simpy_core/             # 시뮬레이션 핵심 모듈
│   ├── __init__.py
│   ├── machine.py          # Machine 클래스
│   ├── scheduler.py        # Scheduler 클래스
│   ├── job.py             # Job 클래스
│   └── data_loader.py     # 데이터 로딩 유틸리티
├── algorithms/             # 알고리즘 모듈들
│   ├── __init__.py
│   ├── genetic/           # 유전 알고리즘
│   │   └── __init__.py
│   └── rule_based/        # 규칙 기반 모델링
│       └── __init__.py
├── requirements.txt       # 의존성 관리
└── README.md
```

