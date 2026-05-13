from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


# PM threshold 후보 값. 작을수록 자주 PM(보수적), 클수록 늦게 PM(공격적).
# 기존 시뮬레이션의 default(0.1, 0.2)를 가운데에 두고 양쪽으로 폭을 줌.
DEFAULT_PM_LEVELS = [0.05, 0.1, 0.2, 0.5, 1.0]


@dataclass
class EncodedData:
    # job_seq 유전자가 가리키는 테이블. 예: ['J1', 'J2', 'J3', ...]
    job_index_table: List[str]

    # pm 유전자가 가리키는 테이블. 예: ['M1', 'M2', 'M3', ...]
    machine_index_table: List[str]

    # machine 유전자의 i번째가 어떤 operation에 해당하는지.
    # 예: ['J1_O1', 'J1_O2', 'J2_O1', ...]  (job_index_table 순서로 정렬됨)
    operation_index_table: List[str]

    # 각 operation별 가능한 머신 리스트.
    # 예: feasible_machine_table[2] = ['M1', 'M2', 'M3']
    #     → 2번째 op는 M1/M2/M3 중 하나에서 처리 가능
    feasible_machine_table: List[List[str]]

    # PM threshold 후보 값. pm 유전자의 값이 이 리스트의 인덱스.
    pm_levels: List[float]


def encode(data: Dict[str, pd.DataFrame]) -> EncodedData:
    """시뮬레이션 데이터(DataFrame들)를 GA가 사용할 인덱스 테이블로 변환.

    GA 시작 시 1번 호출 후 모든 세대에서 재사용.
    """
    # csv 등장 순서를 그대로 사용 → 결정론적
    job_index_table = data['jobs']['job_id'].tolist()
    machine_index_table = data['machines']['machine_id'].tolist()

    # operation은 job_index_table 순서로, 같은 job 내에서는 op_seq 순서로 정렬
    ops_df = data['operations']
    operation_index_table = []
    for job_id in job_index_table:
        job_ops = ops_df[ops_df['job_id'] == job_id].sort_values('op_seq')
        operation_index_table.extend(job_ops['op_id'].tolist())

    # op_id별 가능한 머신 리스트 (DataFrame 한번에 dict으로 변환 → 빠름)
    op_to_machines = (
        data['operation_machine_map']
        .groupby('op_id')['machine_id']
        .apply(list)
        .to_dict()
    )
    feasible_machine_table = [op_to_machines[op_id] for op_id in operation_index_table]

    return EncodedData(
        job_index_table=job_index_table,
        machine_index_table=machine_index_table,
        operation_index_table=operation_index_table,
        feasible_machine_table=feasible_machine_table,
        pm_levels=list(DEFAULT_PM_LEVELS),
    )
