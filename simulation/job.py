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
        self.__qtime[0] = float('inf')
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
        self.__work_start = 0.0
        self.cur_state = Job.State.UNRELEASED
        self.__is_qtime_over = False
        # WAITING 진입 시각. priority._waiting_time(A-2)가 candidate 변별용으로 사용.
        # 아직 RELEASED 전이면 None → priority에서 0으로 취급.
        self.__wait_start_time = None

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
    def total_ops(self):
        """이 job의 총 operation 수. priority.COMPLETION_* 항이 사용."""
        return len(self.__op_seq)

    @property
    def priority(self):
        return self.__priority

    def get_op_group(self):
        """
        현재 operation에 대한 그룹 정보 반환
        """
        return self.__op_group[self.__cur_seq][0]

    def get_next_transition(self):
        """
        '현재 op_group -> 다음 op_group' 문자열을 반환한다 (예: 'G1->G3').
        다음 op이 없으면 None.
        priority._transition_risk에서 c_transition 항 산출에 사용한다.
        """
        nxt = self.__cur_seq + 1
        if nxt >= len(self.__op_group):
            return None
        return f"{self.__op_group[self.__cur_seq][0]}->{self.__op_group[nxt][0]}"

    def get_waiting_time(self):
        """
        현재 WAITING 상태에 머무른 시간 (env.now - 마지막 WAITING 진입 시각).

        같은 op_group candidate라도 stocker에 도착한 시점이 달라
        대기 누적이 달라지므로 candidate 내 variance가 자연스럽게 보장된다.
        priority._waiting_time에서 α 항(A-2: FIFO 일관성)으로 사용한다.
        아직 WAITING에 진입하지 않은 상태면 0.0.
        """
        if self.__wait_start_time is None:
            return 0.0
        return self.__env.now - self.__wait_start_time

    def __chk_qtime(self):
        """
        QTime 체크 프로세스
        """
        try:
            if not self.__is_qtime_over:
                self.__qtime_chk_start = self.__env.now
                remaining = self.__qtime[self.__cur_seq] - self.__waited_time
                if remaining <= 0:
                    self.__qtime_event_idx = self.__event_logger.log_event_start(
                        self.id, 'qtime_over', 'job', self.get_current_operation(), None,
                        at=self.__qtime_chk_start + remaining)
                    self.__is_qtime_over = True
                    return
                yield self.__env.timeout(remaining)
                self.__qtime_event_idx = self.__event_logger.log_event_start(self.id, 'qtime_over', 'job', self.get_current_operation(), None)
                self.__is_qtime_over = True
        except simpy.Interrupt:
            self.__waited_time += self.__env.now - self.__qtime_chk_start

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
        self.__wait_start_time = self.__env.now
        self.__cur_event_idx = self.__event_logger.log_event_start(self.id,
                                                                   'waiting',
                                                                   'job', self.get_current_operation(), None)
        self.__event_queue.put(self)

    def set_state(self, state: State):
        self.cur_state = state
        if state == Job.State.WORKING:
            self.__work_start = self.__env.now
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
            self.__wait_start_time = self.__env.now
            self.__cur_event_idx = self.__event_logger.log_event_start(self.id,
                                                                       f'waiting',
                                                                       'job', self.get_current_operation(), None)
        if is_completed:
            self.__is_qtime_over = False
            self.__waited_time = 0.0
        elif not self.prev_not_completed:
            # working 중 고장(setup 통과 후): 멈춰 있던 qtime 예산에 실패 process 경과시간을 적산
            # setup 중 고장(prev_not_completed=True)이면 qtime 타이머가 계속 돌고 있어 별도 charge 불필요.
            self.__waited_time += self.__env.now - self.__work_start
        self.__event_queue.put(self)
