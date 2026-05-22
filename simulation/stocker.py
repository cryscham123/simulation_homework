from .job import Job
import simpy
import os
import random

class Stocker():
    def __init__(self, env, signal, op_machine=None, job_priority=None):
        self.__env = env
        self.__resource = simpy.FilterStore(env, capacity=float('inf'))
        self.machine_end_signal = signal
        self.__waiting_machines = simpy.FilterStore(env, capacity=float('inf'))
        self.__op_machine = op_machine
        # job_id → 순위 인덱스. 낮을수록 우선. lookup O(1).
        self.__job_priority = (
            {jid: i for i, jid in enumerate(job_priority)} if job_priority is not None else None
        )
        env.process(self.wait_until_machine_ready())

    def __is_match(self, job: Job, machine):
        """GA 모드: op_id가 배정된 정확한 머신만. 룰 모드: 같은 op_group."""
        if self.__op_machine is not None:
            return self.__op_machine[job.get_current_operation()] == machine.id
        return machine.group == job.get_op_group()

    def __pick_job(self, candidates, machine):
        """GA 모드: priority 인덱스 최소. 룰 모드: JOB_RULE env."""
        if self.__job_priority is not None:
            return min(candidates, key=lambda j: self.__job_priority[j.id])
        rule = os.getenv('JOB_RULE', 'random')
        return self.__select_job(candidates, machine, rule)

    def add_job(self, job: Job):
        """
        job을 stocker에 추가.
        매칭되는 대기 중인 machine이 있으면 즉시 dispatch, 없으면 FilterStore에서 대기.
        """
        matching = [
            m for m in self.__waiting_machines.items
            if self.__is_match(job, m)
        ]
        if matching:
            best = min(matching, key=lambda m: int(m.id[1:]))
            machine = yield self.__waiting_machines.get(lambda m: m is best)
            self.__dispatch(job, machine)
        else:
            yield self.__resource.put(job)

    def __dispatch(self, job: Job, machine):
        """job을 machine으로 dispatch하고 관련 프로세스 시작"""
        machine.set_busy(True)
        self.__env.process(machine.run(job))
        self.__env.process(job.operation_completed())

    def __select_job(self, candidates, machine, rule):
        """
        JOB_RULE에 따라 stocker의 candidates 중 하나의 job을 선택
        """
        if rule == 'random':
            return random.choice(candidates)
        if rule == 'FIFO':
            return candidates[0]
        if rule == 'SPT':
            return min(
                candidates,
                key=lambda j: machine.get_process_time(j.get_current_operation())
            )
        if rule == 'LPT':
            return max(
                candidates,
                key=lambda j: machine.get_process_time(j.get_current_operation())
            )
        if rule == 'MIN_QTIME':
            return min(candidates, key=lambda j: j.get_remain_qtime())
        if rule == 'SPTSSU':
            return min(
                candidates,
                key=lambda j: machine.get_setup_time(j.job_type)
                + machine.get_process_time(j.get_current_operation())
            )
        raise ValueError(f"알 수 없는 JOB_RULE 값: {rule}")

    def wait_until_machine_ready(self):
        """
        machine이 idle 신호를 보내면 JOB_RULE에 따라 stocker에서 job 한 개를 선택해 dispatch.
        대기 중인 job이 없으면 machine을 waiting_machines에 추가.
        """
        while True:
            machine = yield self.machine_end_signal.get()
            candidates = [
                x for x in self.__resource.items
                if self.__is_match(x, machine)
            ]
            if len(candidates) == 0:
                yield self.__waiting_machines.put(machine)
                continue
            best = self.__pick_job(candidates, machine)
            job = yield self.__resource.get(lambda x: x is best)
            self.__dispatch(job, machine)
