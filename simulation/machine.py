import simpy
from math import inf
import random
import math
import pandas as pd
from typing import Dict, Any
from utils import EventLogger
from enum import Enum
from .job import Job
import os

class Machine:
    class State(Enum):
        IDLE = 0
        SETUP = 1
        WORKING = 2
        REPAIRING = 3
        PM = 4

    class RepairStatus(Enum):
        SUCCESS_PM = 0
        FAILED_PM = 1

    def __init__(self, env: simpy.Environment, id: str, group: str,
                 failure_info: Dict[str, Any], setup_time_info: pd.DataFrame,
                 process_time_info: pd.DataFrame, pm_hazard_threshold: float,
                 event_logger: EventLogger, event_queue: simpy.Store, machine_signal: simpy.Store):
        """
        Machine 초기화

        Args:
            env: SimPy 환경
            id: 머신 ID
            group: 머신 그룹
            failure_info: 고장 정보 딕셔너리
            setup_time_info: 셋업 시간 정보 DataFrame
            process_time_info: 프로세싱 시간 정보 DataFrame
            pm_hazard_threshold: PM 고장 확률 임계값
            event_logger: 이벤트 기록 인스턴스
            event_queue: 머신 이벤트를 기록할 queue
            machine_signal: 머신 상태 변화 신호를 위한 store
        """
        self.__env = env
        self.__id = id
        self.group = group
        self.__event_logger = event_logger
        self.__event_queue = event_queue

        self.__resource = simpy.PreemptiveResource(env, capacity=1)

        # 고장 관련 파라미터 (Weibull 분포)
        self.__shape = failure_info['shape parameter']
        self.__scale = failure_info['scale parameter']
        self.__repair_time = failure_info['repair_time']
        self.__pm_duration = failure_info['pm_duration']
        self.__pm_hazard_threshold = pm_hazard_threshold

        # 시간 정보
        self.__setup_times = setup_time_info
        self.__process_times = process_time_info

        # 머신 상태
        self.__last_job_type = None
        self.__event_idx = -1
        self.__repair_idx = -1
        self.__PM_idx = -1
        self.cur_state = Machine.State.IDLE
        self.required_state = None
        self.down_process = None
        self.pm_process = None
        self.repair_process = None
        self.machine_signal = machine_signal
        self.preempt = False

    def program_done(self):
        """
        소멸자가 작동 안해서 그냥 명시적으로 머신이 소멸될 때 호출하는 함수 따로 만듦
        """
        self.__event_logger.log_event_finish(self.__event_idx)
        self.__event_logger.log_event_finish(self.__PM_idx)
        self.__event_logger.log_event_finish(self.__repair_idx)

    @property
    def id(self) -> str:
        """머신 ID 반환"""
        return self.__id

    def __calculate_hazard(self):
        """
        Weibull 분포 기반 고장 발생 시간 샘플링.
        """
        if os.getenv('DOWN_ACTIVE', 'True').lower() == 'false':
            return inf
        k = self.__shape
        lam = self.__scale
        if k <= 0 or lam <= 0:
            return inf
        u = random.random()
        return lam * ((-math.log(u)) ** (1.0 / k))

    def down(self):
        """머신 중단 프로세스"""
        try:
            down_time = self.__calculate_hazard()
            yield self.__env.timeout(down_time)
            if self.cur_state in [Machine.State.PM, Machine.State.REPAIRING]:
                return
        except simpy.Interrupt:
            # 예방 보전 성공으로 인한 인터럽트 발생
            return
        self.required_state = Machine.State.REPAIRING
        self.__event_queue.put(self)

    def __calculate_PM_time(self):
        """
        Weibull hazard h(t) = (k/λ)(t/λ)^(k-1)이 threshold에 도달하는 시점.
        h(t) = thr → t = λ · (thr·λ/k)^(1/(k-1))
        """
        if os.getenv('PM_ACTIVE', 'True').lower() == 'false':
            return inf
        k = self.__shape
        lam = self.__scale
        thr = self.__pm_hazard_threshold
        if k <= 0 or lam <= 0:
            return inf
        if k == 1:
            # 지수 분포: hazard 상수 1/λ
            return 0.0 if (1.0 / lam) >= thr else inf
        if k < 1:
            # 감소 hazard: 초기에 가장 높음
            return 0.0
        return lam * ((thr * lam / k) ** (1.0 / (k - 1)))

    def PM(self):
        """예방 보전 프로세스"""
        try:
            yield self.__env.timeout(self.__calculate_PM_time())
            if self.cur_state in [Machine.State.PM, Machine.State.REPAIRING]:
                return
        except simpy.Interrupt:
            # 머신 고장으로 인한 인터럽트 발생
            return
        self.required_state = Machine.State.PM
        self.__event_queue.put(self)

    def repair(self):
        """
        머신 수리 프로세스
        나중에 리팩토링 예정.
        """
        preempt, reason, time = (True, 'repairing', self.__repair_time) if self.required_state == Machine.State.REPAIRING else (False, 'PM', self.__pm_duration)
        try:
            with self.__resource.request(priority=-1, preempt=preempt) as req:
                yield req
                self.set_busy(False)
                self.__last_job_type = None
                if reason == 'PM':
                    self.cur_state = Machine.State.PM
                    self.__PM_idx = self.__event_logger.log_event_start(self.__id, reason, 'machine', None, None)
                else:
                    self.cur_state = Machine.State.REPAIRING
                    self.__repair_idx = self.__event_logger.log_event_start(self.__id, reason, 'machine', None, None)
                yield self.__env.timeout(time)
                # 수리시 setup 정보도 초기화
                self.cur_state = Machine.State.IDLE
                ret = self.RepairStatus.SUCCESS_PM if reason == 'PM' else None
        except simpy.Interrupt:
            ret = self.RepairStatus.FAILED_PM if reason == 'PM' else None
        if reason == 'PM':
            self.__event_logger.log_event_finish(self.__PM_idx)
            self.__PM_idx = -1
        else:
            self.__event_logger.log_event_finish(self.__repair_idx)
            self.__repair_idx = -1
        self.required_state = Machine.State.IDLE
        return ret

    def is_idle(self) -> bool:
        """
        머신이 가용 가능한 상태인지 확인.
        수리중인지 아닌지를 판별하는 용도로 사용
        """
        if self.preempt:
            return False
        return self.__resource.count < self.__resource.capacity

    def set_busy(self, status):
        self.preempt = status

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

    def run(self, job):
        """
        머신의 메인 프로세스

        Args:
            job: 머신에서 처리할 작업
        """
        is_completed = False
        job.prev_not_completed = True
        try:
            with self.__resource.request(priority=0, preempt=False) as req:
                yield req
                self.set_busy(False)
                op_id = job.get_current_operation()

                self.cur_state = Machine.State.SETUP
                job.set_state(Job.State.SETUP)
                self.__event_idx = self.__event_logger.log_event_start(self.__id, 
                                                                       'setup', 
                                                                       'machine', op_id,
                                                                       f'job: {job.id}\noperation: {op_id}')
                yield self.__env.timeout(self.get_setup_time(job.job_type))
                self.__last_job_type = job.job_type
                self.__event_logger.log_event_finish(self.__event_idx)

                job.interrupt_qtime()
                job.prev_not_completed = False

                self.cur_state = Machine.State.WORKING
                job.set_state(Job.State.WORKING)
                self.__event_idx = self.__event_logger.log_event_start(self.__id, 
                                                                       f'working-{job.job_type}', 
                                                                       'machine', op_id,
                                                                       f'job: {job.id}\noperation: {op_id}')
                yield self.__env.timeout(self.get_process_time(op_id))

                self.cur_state = Machine.State.IDLE
                is_completed = True
        except simpy.Interrupt:
            pass
        self.__event_logger.log_event_finish(self.__event_idx)
        if job is not None:
            job.operation_end_signal.put(is_completed)
        self.__event_idx = -1
        self.machine_signal.put(self)
