import simpy
import pandas as pd
from .machine import Machine
from utils import EventLogger
from typing import Dict
from .job import Job
from .stocker import Stocker

class Scheduler:
    """시뮬레이션 환경의 스케줄러 클래스"""

    def __init__(self, 
                 env: simpy.Environment, 
                 data: Dict[str, pd.DataFrame], 
                 event_logger: EventLogger, 
                 pm_hazard_threshold: float):
        """
        Scheduler 초기화

        Args:
            env: SimPy 환경
            data: 시뮬레이션에 필요한 데이터 딕셔너리
            event_logger: 이벤트 기록 인스턴스
            pm_hazard_threshold: PM 고장 확률 임계값
        """
        self.__env = env
        self.__WIP = 0
        self.__machines = []
        self.machine_signal = simpy.Store(env, capacity=float('inf'))
        self.machine_events = simpy.Store(env, capacity=float('inf'))

        # 머신 인스턴스 생성 및 스토어에 추가
        for machine_id, row in data['machines'].set_index('machine_id').iterrows():
            machine_group = row['machine_group']

            # 해당 머신의 고장 정보 가져오기
            machine_failure_df = data['machine_failure']
            failure_info = machine_failure_df[
                machine_failure_df['machine_id'] == machine_id
            ].iloc[0].to_dict()

            # 해당 머신 그룹의 셋업 시간 정보 가져오기
            setup_times_df = data['setup_times']
            setup_time_info = setup_times_df[
                setup_times_df['machine_group'] == machine_group
            ]

            # 해당 머신의 처리 시간 정보 가져오기
            op_machine_df = data['operation_machine_map']
            process_time_info = op_machine_df[
                op_machine_df['machine_id'] == machine_id
            ]

            machine = Machine(
                env=env,
                id=machine_id,
                group=machine_group,
                failure_info=failure_info,
                setup_time_info=setup_time_info,
                process_time_info=process_time_info,
                pm_hazard_threshold=pm_hazard_threshold,
                event_logger=event_logger,
                event_queue=self.machine_events,
                machine_signal=self.machine_signal,
            )
            machine.down_process = env.process(machine.down())
            machine.pm_process = env.process(machine.PM())
            self.__machines.append(machine)
        self.__stocker = Stocker(env, self.machine_signal)
        # 시뮬레이션 시작 시 모든 machine의 초기 idle 상태를 stocker에 알림
        for machine in self.__machines:
            self.machine_signal.put(machine)
        env.process(self.__chk_machine_event())

        self.__jobs = []
        self.job_events = simpy.Store(env, capacity=float('inf'))
        for _, job_info in data['jobs'].iterrows():
            # 해당 작업의 operation 정보 가져오기
            job_operations = data['operations'].loc[
                data['operations']['job_id'] == job_info['job_id'],
            ].sort_values('op_seq')

            job = Job(
                env=env,
                job_info=job_info.to_dict(),
                op_info=job_operations,
                event_logger=event_logger,
                event_queue=self.job_events
            )
            self.__jobs.append(job)
            env.process(job.release())
        self.job_chk_process = env.process(self.__chk_job_waiting(len(self.__jobs)))

    def __chk_machine_event(self):
        """
        머신 고장 체크 프로세스
        """
        while True:
            machine = yield self.machine_events.get()
            if machine.required_state == Machine.State.REPAIRING:
                if machine.cur_state == Machine.State.PM:
                    continue
                if machine.pm_process.is_alive:
                    machine.pm_process.interrupt()
                if machine.repair_process is not None and machine.repair_process.is_alive:
                    machine.repair_process.interrupt()
            self.__env.process(self.__repair_and_reschedule_machine(machine))

    def __repair_and_reschedule_machine(self, machine: Machine):
        """
        머신 수리 프로세스

        Args:
            machine: 수리할 머신
        """
        machine.repair_process = self.__env.process(machine.repair())
        status = yield machine.repair_process
        # PM에 성공하면 머신 고장 확률 초기화
        if status == Machine.RepairStatus.SUCCESS_PM:
            if machine.down_process.is_alive:
                machine.down_process.interrupt()
        elif status == Machine.RepairStatus.FAILED_PM:
            return
        machine.down_process = self.__env.process(machine.down())
        machine.pm_process = self.__env.process(machine.PM())
        self.machine_signal.put(machine)

    def __chk_job_waiting(self, num_jobs: int):
        """
        작업 대기 체크 프로세스
        """
        terminated_jobs = 0
        while terminated_jobs < num_jobs:
            job = yield self.job_events.get()
            self.__WIP += (job.cur_state == Job.State.RELEASED) - (job.cur_state == Job.State.COMPLETED)
            if job.cur_state == Job.State.COMPLETED:
                terminated_jobs += 1
                continue
            # 작업 대기 상태 혹은 대기, 세팅, 작업 도중 기계 고장 시 다시 매칭 시도
            self.__matching_machine(job)
        for machine in self.__machines:
            machine.program_done()
        for job in self.__jobs:
            job.program_done()

    def __matching_machine(self, job: Job):
        """
        작업과 매칭되는 머신을 찾아 작업 실행 프로세스 시작.
        모든 job은 항상 stocker를 경유하며, dispatch는 JOB_RULE에 의해 결정된다.

        Args:
            job: 매칭할 작업
        """
        if not job.prev_not_completed:
            job.start_qtime_chk()
        self.__env.process(self.__stocker.add_job(job))
