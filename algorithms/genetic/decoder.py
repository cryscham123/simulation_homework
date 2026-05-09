from dataclasses import dataclass
from typing import Dict, List

from .chromosome import Chromosome
from .encoder import EncodedData


@dataclass
class SimulationInput:
    """GA가 결정한 의사결정을 시뮬레이터에 주입하는 데이터 묶음."""

    # 1순위 → 마지막 순위로 정렬된 job_id 리스트.
    # 예: ['J4', 'J1', 'J3', ...] (J4가 1순위)
    job_priority: List[str]

    # op_id → 그 op를 처리할 machine_id.
    # 예: {'J1_O1': 'M2', 'J1_O2': 'M4', ...}
    op_machine: Dict[str, str]

    # machine_id → 그 머신의 PM threshold 값.
    # 예: {'M1': 0.2, 'M2': 0.1, ...}
    pm_thresholds: Dict[str, float]


def decode(chromo: Chromosome, encoded: EncodedData) -> SimulationInput:
    """염색체를 시뮬레이션 입력 형태로 변환."""
    # job_seq[i] = g → i+1 순위로 투입할 job은 job_index_table[g]
    job_priority = [encoded.job_index_table[g] for g in chromo.job_seq]

    # machine[i] = j → i번 op는 feasible_machine_table[i][j] 머신에서 처리
    op_machine = {
        encoded.operation_index_table[i]: encoded.feasible_machine_table[i][gene]
        for i, gene in enumerate(chromo.machine)
    }

    # pm[m] = k → m번 머신의 threshold는 pm_levels[k]
    pm_thresholds = {
        encoded.machine_index_table[m]: encoded.pm_levels[gene]
        for m, gene in enumerate(chromo.pm)
    }

    return SimulationInput(
        job_priority=job_priority,
        op_machine=op_machine,
        pm_thresholds=pm_thresholds,
    )
