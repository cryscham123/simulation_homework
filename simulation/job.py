import simpy
import pandas as pd
from typing import Dict, Any
from enum import Enum
from utils import EventLogger

class Job:
    class State(Enum):
        UNRELEASED = 0
        RELEASED = 1
        WAITING = 2
        SETUP = 3
        WORKING = 4
        COMPLETED = 5

    def __init__(self, env: simpy.Environment, job_info: Dict[str, Any],
                 op_info: pd.DataFrame, event_logger: EventLogger, event_queue: simpy.Store):
        """
        Job 초기화

        Args:
            env: SimPy 환경
            job_info: 작업 정보 딕셔너리
            op_info: 작업 operation 정보 DataFrame
            event_logger: 이벤트 기록 인스턴스
            event_queue: 작업 이벤트를 기록할 queue
        """
        self.__env = env
        self.__id = job_info['job_id']
        self.__event_logger = event_logger
        self.__event_queue = event_queue
        self.__job_type = job_info['job_type']
        self.__release_time = job_info['release_time']
        self.__due_date = job_info['due_date']
        self.__priority = job_info['priority']
        self.__qtime = op_info['qtime'].astype(float).values
        self.__qtime[self.__qtime <= 0] = float('inf')
        self.__op_seq = op_info[['op_id', 'op_seq']].values
        self.__op_group = op_info[['op_group', 'op_seq']].values

        # 프로세스 상태 관리
        self.__cur_seq = 0
        self.__qtime_process = None
        self.__cur_event_idx = -1
        self.__qtime_event_idx = -1
        self.operation_end_signal = simpy.Store(env)
        self.__qtime_chk_start = 0.0
        self.__waited_time = 0.0
        self.cur_state = Job.State.UNRELEASED
        self.__is_qtime_over = False

        self.prev_not_completed = False

    def program_done(self):
        """
        소멸자가 동작 안해서 명시적으로 프로그램 종료 함수 만듦
        정확히 측정되지 않은 qtime 오버시간은 추가 반영 안함.
        """
        self.__event_logger.log_event_finish(self.__cur_event_idx)
        self.__event_logger.log_event_finish(self.__qtime_event_idx)

    @property
    def id(self):
        return self.__id

    @property
    def job_type(self):
        return self.__job_type

    @property
    def cur_seq(self):
        return self.__cur_seq

    @property
    def priority(self):
        return self.__priority

    def get_op_group(self):
        """
        현재 operation에 대한 그룹 정보 반환
        """
        return self.__op_group[self.__cur_seq][0]

    def __chk_qtime(self):
        """
        QTime 체크 프로세스
        """
        try:
            if not self.__is_qtime_over:
                self.__qtime_chk_start = self.__env.now
                yield self.__env.timeout(self.__qtime[self.__cur_seq] - self.__waited_time)
                self.__qtime_event_idx = self.__event_logger.log_event_start(self.id, 'qtime_over', 'job', self.get_current_operation(), None)
                self.__is_qtime_over = True
        except simpy.Interrupt:
            self.__waited_time = self.__env.now - self.__qtime_chk_start

    def start_qtime_chk(self):
        """
        QTime 체크 프로세스 시작
        """
        self.__qtime_process = self.__env.process(self.__chk_qtime())

    def interrupt_qtime(self):
        """
        QTime 체크 프로세스 중단
        """
        if self.__qtime_process.is_alive:
            self.__qtime_process.interrupt()
            return
        self.__event_logger.log_event_finish(self.__qtime_event_idx)

    def get_remain_qtime(self):
        """
        남은 QTime 반환. 음수일 경우 QTime 초과 상태
        """
        return (self.__qtime[self.__cur_seq] - self.__waited_time) - (self.__env.now - self.__qtime_chk_start)

    def get_current_operation(self):
        if self.cur_state in [self.State.COMPLETED, self.State.UNRELEASED]:
            return None
        return self.__op_seq[self.__cur_seq][0]

    def release(self):
        yield self.__env.timeout(self.__release_time)
        self.cur_state = self.State.RELEASED
        self.__cur_event_idx = self.__event_logger.log_event_start(self.id, 
                                                                   'waiting', 
                                                                   'job', self.get_current_operation(), None)
        self.__event_queue.put(self)

    def set_state(self, state: State):
        self.cur_state = state
        self.__event_logger.log_event_finish(self.__cur_event_idx)
        self.__cur_event_idx = self.__event_logger.log_event_start(self.id, 
                                                                   'setup' if state == Job.State.SETUP else 'working', 
                                                                   'job', self.get_current_operation(), None)

    def operation_completed(self):
        is_completed = yield self.operation_end_signal.get()
        self.__cur_seq += int(is_completed)
        self.__event_logger.log_event_finish(self.__cur_event_idx)
        if self.__cur_seq >= len(self.__op_seq):
            self.cur_state = self.State.COMPLETED
            self.__cur_event_idx = -1
            self.__qtime_event_idx = -1
        else:
            self.cur_state = self.State.WAITING
            self.__cur_event_idx = self.__event_logger.log_event_start(self.id, 
                                                                       f'waiting-{self.__job_type}', 
                                                                       'job', self.get_current_operation(), None)
        self.__is_qtime_over &= (not is_completed)
        self.__waited_time *= int(not is_completed)
        self.__event_queue.put(self)
