import random
from typing import Dict, Tuple

import pandas as pd
import simpy

from utils import EventLogger
from simulation import Scheduler

from .chromosome import Chromosome
from .decoder import decode
from .encoder import EncodedData


class Evaluator:
    """Chromosome 1개를 시뮬레이션으로 평가해 (makespan, qtime_violation) 반환."""

    def __init__(self,
                 encoded: EncodedData,
                 data: Dict[str, pd.DataFrame],
                 seed: int = 42):
        self.encoded = encoded
        self.data = data
        self.seed = seed

    def evaluate(self, chromo: Chromosome) -> Tuple[float, float]:
        """염색체 1개를 시뮬해서 (makespan, qtime_violation) 반환. fitness는 호출자가 채움."""
        # 모든 chromosome이 같은 랜덤 환경에서 평가받도록 매번 시드 reset
        random.seed(self.seed)

        sim_input = decode(chromo, self.encoded)

        env = simpy.Environment()
        event_logger = EventLogger(env)
        scheduler = Scheduler(
            env=env,
            data=self.data,
            event_logger=event_logger,
            pm_hazard_threshold=0.0,  # GA에선 pm_thresholds가 우선이라 안 쓰임 값을 넘기기는 해야해서 임의로
            job_priority=sim_input.job_priority,
            op_machine=sim_input.op_machine,
            pm_thresholds=sim_input.pm_thresholds,
        )
        env.run(until=scheduler.job_chk_process)

        return self._compute_metrics(event_logger)

    def _compute_metrics(self, event_logger: EventLogger) -> Tuple[float, float]:
        df = pd.DataFrame(event_logger.logs)
        job_info = df[df['resource'] == 'job'].copy()

        # makespan: 마지막 job 종료 시점
        makespan = float(job_info['finish'].max())

        # qtime_violation: 모든 qtime_over 이벤트의 duration 합계
        job_info['duration'] = job_info['finish'] - job_info['start']
        qtime_violation = float(
            job_info.loc[job_info['event'] == 'qtime_over', 'duration'].sum()
        )

        return makespan, qtime_violation
