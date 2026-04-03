import simpy
import random
import math
import pandas as pd
from typing import Dict, Any


class Machine:
    def __init__(self, env: simpy.Environment, id: int, group: str,
                 failure_info: Dict[str, Any], setup_time_info: pd.DataFrame,
                 process_time_info: pd.DataFrame):
        """
        Machine 초기화

        Args:
            env: SimPy 환경
            id: 머신 ID
            group: 머신 그룹
            failure_info: 고장 정보 딕셔너리
            setup_time_info: 셋업 시간 정보 DataFrame
            process_time_info: 프로세싱 시간 정보 DataFrame
        """
        self.__env = env
        self.__id = id
        self.group = group

        # 가정: Machine은 한 번에 하나의 작업만 처리할 수 있다.
        self.resource = simpy.PreemptiveResource(env, capacity=1)

        # 고장 관련 파라미터
        self.__base_hazard = failure_info['base_hazard']
        self.__hazard_increase_rate = failure_info['hazard_increase_rate']
        self.__repair_time = failure_info['repair_time']
        # PM 관련 로직은 현재 구현되어 있지 않다.
        self.__pm_duration = failure_info['pm_duration']

        # 시간 정보
        self.__setup_times = setup_time_info
        self.__process_times = process_time_info

        # 머신 상태
        self.__is_repaired = True
        self.__last_repair_time = 0.0
        self.__last_job_type = None

        # 고장 프로세스 시작
        env.process(self.__breakdown())

    @property
    def id(self) -> int:
        """머신 ID 반환"""
        return self.__id

    def __calculate_hazard(self):
        # h(t) = h0 + hr*t (시간에 따라 증가하는 위험률)
        # H(t) = h0*t + hr*t²/2 (누적 위험 함수)
        # H(T) = -ln(U)를 만족하는 T를 계산 (U ~ Uniform(0,1))
        # h0*T + hr*T²/2 = -ln(U)
        # 2차 방정식: hr*T²/2 + h0*T + ln(U) = 0
        # 해: T = (-h0 + sqrt(h0² - 2*hr*ln(U))) / hr

        h0 = self.__base_hazard
        hr = self.__hazard_increase_rate
        u = random.random()

        return (-h0 + math.sqrt(h0**2 - 2*hr*math.log(u))) / hr


    def __breakdown(self):
        """머신 고장 프로세스"""
        while True:
            yield self.__env.timeout(self.__calculate_hazard())

            # 아직 수리가 안끝났는데 또 고장 이벤트가 발생하면 넘어감
            if not self.__is_repaired:
                continue

            self.__is_repaired = False
            # priority를 -1, preemt=True로 둠으로써, 현재 다른 작업을 수행중이라도 고장 이벤트가 우선적으로 발생하도록 설계
            with self.resource.request(priority=-1, preempt=True) as req:
                yield req
                print(f'{round(self.__env.now, 2)}\tMachine {self.__id} broke down')
                yield self.__env.process(self.__repair())
            self.__is_repaired = True

    def __repair(self):
        """머신 수리 프로세스"""
        yield self.__env.timeout(self.__repair_time)
        self.__last_repair_time = self.__env.now
        # 수리시 setup 정보도 초기화
        self.__last_job_type = None
        print(f'{round(self.__env.now, 2)}\tMachine {self.__id} repaired')

    def is_idle(self) -> bool:
        """
        머신이 가용 가능한 상태인지 확인.
        수리중인지 아닌지를 판별하는 용도로 사용
        """
        return self.resource.count < self.resource.capacity

    def get_setup_time(self, job_type: str) -> float:
        """
        주어진 작업 타입에 대한 셋업 시간 반환

        Args:
            job_type: 작업 타입

        Returns:
            float: 셋업 시간
        """
        # 이전 작업이 없을 경우 setup time은 없다고 가정
        if self.__last_job_type is None or job_type == self.__last_job_type:
            return 0

        setup_time_row = self.__setup_times[
            (self.__setup_times['from_job_type'] == self.__last_job_type) &
            (self.__setup_times['to_job_type'] == job_type)
        ]
        return setup_time_row['setup_time'].iloc[0]

    def get_process_time(self, op_id: int) -> float:
        """
        주어진 작업 ID에 대한 처리 시간 반환

        Args:
            op_id: 작업 ID

        Returns:
            float: 처리 시간
        """
        process_time_row = self.__process_times[
            self.__process_times['op_id'] == op_id
        ]
        return process_time_row['process_time'].iloc[0]

    def setup(self, job_type: str):
        """
        머신 셋업 프로세스

        Args:
            job_type: 작업 타입
        """
        yield self.__env.timeout(self.get_setup_time(job_type))
        self.__last_job_type = job_type

    def work(self, op_id: int):
        """
        작업 처리 프로세스

        Args:
            op_id: 작업 ID
        """
        yield self.__env.timeout(self.get_process_time(op_id))

