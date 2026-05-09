import pandas as pd
import os
from typing import Dict


class DataLoader:
    def __init__(self, base_data_path: str = ""):
        """
        DataLoader 초기화

        Args:
            base_data_path: 데이터 파일들의 기본 경로
        """
        self.base_data_path = base_data_path

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        모든 데이터를 로드하여 딕셔너리로 반환

        Returns:
            Dict[str, pd.DataFrame]: 각 데이터의 이름을 키로 하는 DataFrame 딕셔너리
        """
        data = {}

        data['machines'] = pd.read_csv(os.path.join(self.base_data_path, 'machines.csv'))
        data['jobs'] = pd.read_csv(os.path.join(self.base_data_path, 'jobs.csv'))
        data['machine_failure'] = pd.read_csv(os.path.join(self.base_data_path, 'machine_failure.csv'))
        data['operation_machine_map'] = pd.read_csv(os.path.join(self.base_data_path, 'operation_machine_map.csv'))
        data['operations'] = pd.read_csv(os.path.join(self.base_data_path, 'operations.csv'))
        data['setup_times'] = pd.read_csv(os.path.join(self.base_data_path, 'setup_times.csv'))

        # Weibull 파라미터:
        #   shape parameter: 무차원 (k) — 변환 불필요
        #   scale parameter: 시간 단위 (λ) — 데이터가 분 단위이므로 시뮬레이션 내부 단위(분)와 일치, 변환 불필요
        DOWN_TIME_UNIT = os.getenv('DOWN_TIME_UNIT', 'M')
        time_constants = {
            'M': 1,
            'H': 60,
            'D': 60 * 24
        }
        data['machine_failure']['scale parameter'] = (
            data['machine_failure']['scale parameter'] * time_constants[DOWN_TIME_UNIT]
        )

        return data
