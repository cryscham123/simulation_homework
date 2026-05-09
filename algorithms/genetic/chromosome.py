from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Chromosome:
    # 길이 = job 개수. 값 = job_index_table의 인덱스.
    # 예: job_seq=[3,0,2,1] → 투입 순서: job_index_table[3], [0], [2], [1]
    job_seq: List[int]

    # 길이 = 전체 operation 개수. 값 = feasible_machine_table[i]의 인덱스.
    # 예: machine[2]=1 → i=2번 op는 feasible_machine_table[2][1] 머신에서 처리
    machine: List[int]

    # 길이 = 머신 개수. 값 = PM_LEVELS의 인덱스 (encoder.py에서 정의).
    # 예: pm[0]=2 → 0번 머신의 PM threshold는 PM_LEVELS[2]
    pm: List[int]

    # 평가 전 None, 평가 후 (makespan, qtime_violation). 둘 다 작을수록 좋음.
    fitness: Optional[Tuple[float, float]] = None
