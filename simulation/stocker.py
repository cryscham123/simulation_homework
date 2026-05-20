from .job import Job
import simpy
import os
import random

class Stocker():
    def __init__(self, env, signal):
        self.__env = env
        self.__resource = simpy.FilterStore(env, capacity=float('inf'))
        self.machine_end_signal = signal
        self.__waiting_machines = simpy.FilterStore(env, capacity=float('inf'))
        env.process(self.wait_until_machine_ready())

    def add_job(self, job: Job):
        """
        job을 stocker에 추가.
        같은 group의 대기 중인 machine이 있으면 즉시 dispatch, 없으면 FilterStore에서 대기.
        """
        matching = [
            m for m in self.__waiting_machines.items
            if m.group == job.get_op_group()
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
                if x.get_op_group() == machine.group
            ]
            if len(candidates) == 0:
                yield self.__waiting_machines.put(machine)
                continue
            rule = os.getenv('JOB_RULE', 'random')
            best = self.__select_job(candidates, machine, rule)
            job = yield self.__resource.get(lambda x: x is best)
            self.__dispatch(job, machine)
