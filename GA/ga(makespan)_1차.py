import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

#유전알고리즘 파라미터
POP_SIZE = 200 #세대당 해 
GENERATIONS = 200 #세대 수(n세대까지 반복)
TOURNAMENT_K = 5 #부모 선택 시 n개 뽑아서 좋은해 선택
CROSSOVER_RATE = 0.9 #확률로 교차 수행
MUTATION_RATE_SEQ = 0.20 #순서 돌연변이
MUTATION_RATE_MAC = 0.45 #장비 선택 돌연변이
ELITE_SIZE = 3 #가장 좋은 해 n개는 다음 세대로 복사
RANDOM_SEED = 42

# 목점함수 가중치
W_MAKESPAN = 1.0
W_QTIME = 1000.0


# Individual class 하나가 schedule 하나(개체가 schedule하나라고 이해하면 됨/ 일종의 schedule후보)
@dataclass
class Individual:
    seq_gene: List[str] #랜덤 job 순서 유전자
    machine_gene: Dict[str, str] #장비 선택 유전자  
    fitness: float = float("inf")
    decoded: dict = None


class GAScheduler:
    def __init__(self,
                 jobs_path='schema/jobs.csv',
                 machines_path='schema/machines.csv',
                 operations_path='schema/operations.csv',
                 op_machine_map_path='schema/operation_machine_map.csv',
                 qtime_path='schema/qtime_constraints.csv',
                 setup_path='schema/setup_times.csv'):
        random.seed(RANDOM_SEED)

        self.jobs = pd.read_csv(jobs_path)
        self.machines = pd.read_csv(machines_path)
        self.operations = pd.read_csv(operations_path)
        self.op_machine_map = pd.read_csv(op_machine_map_path)
        self.qtimes = pd.read_csv(qtime_path)
        self.setup = pd.read_csv(setup_path)

        self._build_lookup()

    #데이터 전처리(스키마 데이터 dictionary로 변환)
    def _build_lookup(self):
        self.job_info = self.jobs.set_index('job_id').to_dict('index')
        self.job_ids = self.jobs['job_id'].tolist()
        self.job_type_of = {r['job_id']: r['job_type'] for _, r in self.jobs.iterrows()}
        self.release_of = {r['job_id']: float(r['release_time']) for _, r in self.jobs.iterrows()}
        self.due_of = {r['job_id']: float(r['due_date']) for _, r in self.jobs.iterrows()}
        self.priority_of = {r['job_id']: float(r['priority']) for _, r in self.jobs.iterrows()}

       
        self.operations = self.operations.sort_values(['job_id', 'op_seq']).reset_index(drop=True)
        self.ops_by_job: Dict[str, List[dict]] = {}
        self.op_seq_of: Dict[str, int] = {}
        self.op_group_of: Dict[str, str] = {}
        self.op_job_of: Dict[str, str] = {}

        for _, row in self.operations.iterrows():
            d = row.to_dict()
            self.ops_by_job.setdefault(row['job_id'], []).append(d)
            self.op_seq_of[row['op_id']] = int(row['op_seq'])
            self.op_group_of[row['op_id']] = row['op_group']
            self.op_job_of[row['op_id']] = row['job_id']

        self.total_num_ops = len(self.operations)
        self.num_ops_per_job = {j: len(v) for j, v in self.ops_by_job.items()}

       
        self.machine_group_of = {r['machine_id']: r['machine_group'] for _, r in self.machines.iterrows()}
        self.machine_ids = self.machines['machine_id'].tolist()

        
        self.feasible_machines: Dict[str, List[str]] = {}
        self.proc_time: Dict[Tuple[str, str], float] = {}
        for _, row in self.op_machine_map.iterrows():
            op_id = row['op_id']
            m = row['machine_id']
            p = float(row['process_time'])
            self.feasible_machines.setdefault(op_id, []).append(m)
            self.proc_time[(op_id, m)] = p

        self.qtime_limit = {}
        for _, row in self.qtimes.iterrows():
            key = (row['job_id'], int(row['from_op_seq']), int(row['to_op_seq']))
            self.qtime_limit[key] = float(row['max_qtime'])

        self.setup_time = {}
        for _, row in self.setup.iterrows():
            key = (row['machine_group'], row['from_job_type'], row['to_job_type'])
            self.setup_time[key] = float(row['setup_time'])

        self.op_id_by_job_seq = {}
        for job_id, rows in self.ops_by_job.items():
            for r in rows:
                self.op_id_by_job_seq[(job_id, int(r['op_seq']))] = r['op_id']

    #염색체 생성
    def create_random_individual(self) -> Individual:
        # sequence gene: job의 operation수 만큼 반복해서 저장
        seq_gene = []
        for j in self.job_ids:
            seq_gene.extend([j] * self.num_ops_per_job[j])
        random.shuffle(seq_gene)

        # machine gene: 해당 operation에 가능한 machine 중 하나를 랜덤 선택
        machine_gene = {}
        for op_id in self.operations['op_id'].tolist():
            machine_gene[op_id] = random.choice(self.feasible_machines[op_id])

        ind = Individual(seq_gene=seq_gene, machine_gene=machine_gene)
        self.evaluate(ind)
        return ind

    # 디코딩
    def decode(self, ind: Individual) -> dict:
        next_op_index = {j: 0 for j in self.job_ids}  #job j의 다음에 배치할 operation 번호
        job_ready_time = {j: self.release_of[j] for j in self.job_ids} #job j가 다음 공정을 시작할 수 있는 가장 빠른 시간(초기는 release_time)
        prev_op_end = {} #이전 operation의 종료시간 저장                                

        machine_ready_time = {m: 0.0 for m in self.machine_ids} #macine m이 다음 작업 시작할 수 있는 시간
        machine_last_job_type = {m: None for m in self.machine_ids} #machine m에서 직전에 처리한 job_type

        schedule_rows = []
        qtime_rows = []

        # job의 몇 번째 operation 배치할지 확인
        for job_id in ind.seq_gene:
            op_idx = next_op_index[job_id]
            if op_idx >= self.num_ops_per_job[job_id]:
                continue 
            
            # 현재 operation 확인
            op = self.ops_by_job[job_id][op_idx]
            op_id = op['op_id']
            op_seq = int(op['op_seq'])
            machine_id = ind.machine_gene[op_id]

            # machine feasible 확인
            if machine_id not in self.feasible_machines[op_id]:
                machine_id = random.choice(self.feasible_machines[op_id])
                ind.machine_gene[op_id] = machine_id

            #setup time 계산
            machine_group = self.machine_group_of[machine_id]
            curr_job_type = self.job_type_of[job_id]
            prev_type = machine_last_job_type[machine_id]
  
            setup = 0.0
            if prev_type is not None:
                setup = self.setup_time.get((machine_group, prev_type, curr_job_type), 0.0)

            earliest_machine_start = machine_ready_time[machine_id] + setup
            earliest_job_start = job_ready_time[job_id]
            start_time = max(earliest_machine_start, earliest_job_start)
            process_time = self.proc_time[(op_id, machine_id)]
            end_time = start_time + process_time

            q_violation = 0.0
            wait_time = 0.0
            if op_seq >= 2:
                prev_end = prev_op_end[(job_id, op_seq - 1)]
                wait_time = start_time - prev_end
                limit = self.qtime_limit.get((job_id, op_seq - 1, op_seq), None)
                if limit is not None:
                    q_violation = max(0.0, wait_time - limit)

            schedule_rows.append({
                'job_id': job_id,
                'op_id': op_id,
                'op_seq': op_seq,
                'machine_id': machine_id,
                'machine_group': machine_group,
                'job_type': curr_job_type,
                'setup_time': setup,
                'process_time': process_time,
                'start_time': start_time,
                'end_time': end_time,
            })

            if op_seq >= 2:
                qtime_rows.append({
                    'job_id': job_id,
                    'from_op_seq': op_seq - 1,
                    'to_op_seq': op_seq,
                    'wait_time': wait_time,
                    'max_qtime': self.qtime_limit.get((job_id, op_seq - 1, op_seq), None),
                    'qtime_violation': q_violation,
                })

            prev_op_end[(job_id, op_seq)] = end_time
            job_ready_time[job_id] = end_time
            machine_ready_time[machine_id] = end_time
            machine_last_job_type[machine_id] = curr_job_type
            next_op_index[job_id] += 1

        completion_time = {j: 0.0 for j in self.job_ids}
        for row in schedule_rows:
            completion_time[row['job_id']] = max(completion_time[row['job_id']], row['end_time'])


        makespan = max(completion_time.values()) if completion_time else 0.0
        total_qtime_violation = sum(r['qtime_violation'] for r in qtime_rows)

        return {
            'schedule_df': pd.DataFrame(schedule_rows).sort_values(['start_time', 'machine_id']).reset_index(drop=True),
            'qtime_df': pd.DataFrame(qtime_rows),
            'makespan': makespan,
            'total_qtime_violation': total_qtime_violation,
        }

    # 평가
    def evaluate(self, ind: Individual) -> float:
        decoded = self.decode(ind)
        fitness = (
            W_MAKESPAN * decoded['makespan']
            + W_QTIME * decoded['total_qtime_violation']
        )
        ind.fitness = fitness
        ind.decoded = decoded
        return fitness

    # 부모 선택
    def tournament_selection(self, population: List[Individual]) -> Individual:
        sampled = random.sample(population, TOURNAMENT_K)
        sampled.sort(key=lambda x: x.fitness)
        return sampled[0]

    # 교차 선택(sequence gene)
    def crossover_sequence(self, p1: List[str], p2: List[str]) -> List[str]:
        size = len(p1)
        mask = [random.randint(0, 1) for _ in range(size)]
        child = [None] * size

        required_counts = {j: self.num_ops_per_job[j] for j in self.job_ids}
        used_counts = {j: 0 for j in self.job_ids}

        for i in range(size):
            if mask[i] == 1:
                job = p1[i]
                if used_counts[job] < required_counts[job]:
                    child[i] = job
                    used_counts[job] += 1

        p2_iter = []
        temp_counts = used_counts.copy()
        for job in p2:
            if temp_counts[job] < required_counts[job]:
                p2_iter.append(job)
                temp_counts[job] += 1

        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_iter[idx]
                idx += 1

        return child

    # 교차 선택(machine gene)
    def crossover_machine_gene(self, g1: Dict[str, str], g2: Dict[str, str]) -> Dict[str, str]:
        child = {}
        for op_id in g1.keys():
            chosen = g1[op_id] if random.random() < 0.5 else g2[op_id]
            if chosen not in self.feasible_machines[op_id]:
                chosen = random.choice(self.feasible_machines[op_id])
            child[op_id] = chosen
        return child

    # 개체(스케줄) 두 개를 받아 자식 두개 생성(부모를 복사 후 교차)
    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > CROSSOVER_RATE:
            c1 = Individual(seq_gene=p1.seq_gene[:], machine_gene=p1.machine_gene.copy())
            c2 = Individual(seq_gene=p2.seq_gene[:], machine_gene=p2.machine_gene.copy())
            return c1, c2

        c1_seq = self.crossover_sequence(p1.seq_gene, p2.seq_gene)
        c2_seq = self.crossover_sequence(p2.seq_gene, p1.seq_gene)
        c1_mac = self.crossover_machine_gene(p1.machine_gene, p2.machine_gene)
        c2_mac = self.crossover_machine_gene(p2.machine_gene, p1.machine_gene)

        return Individual(c1_seq, c1_mac), Individual(c2_seq, c2_mac)

    # 개체 하나에 돌연변이 적용
    def mutate(self, ind: Individual):
        # sequence 돌연변이 : sequence gene 안의 i번째 칸과 j번째 칸 값 swap
        if random.random() < MUTATION_RATE_SEQ:
            i, j = random.sample(range(len(ind.seq_gene)), 2)
            ind.seq_gene[i], ind.seq_gene[j] = ind.seq_gene[j], ind.seq_gene[i]

        # machine 돌연변이 : machine 할당 랜덤하게 수정(feasible한 machine중)
        if random.random() < MUTATION_RATE_MAC:
            num_changes = random.randint(1, max(1, self.total_num_ops // 10))
            op_ids = random.sample(self.operations['op_id'].tolist(), num_changes)
            for op_id in op_ids:
                ind.machine_gene[op_id] = random.choice(self.feasible_machines[op_id])

    # GA 실행
    def run(self):
        population = [self.create_random_individual() for _ in range(POP_SIZE)]
        history = []

        best = min(population, key=lambda x: x.fitness)

        for gen in range(GENERATIONS):
            population.sort(key=lambda x: x.fitness)
            new_population = population[:ELITE_SIZE]

            while len(new_population) < POP_SIZE:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                self.evaluate(c1)
                self.evaluate(c2)
                new_population.append(c1)
                if len(new_population) < POP_SIZE:
                    new_population.append(c2)

            population = new_population
            current_best = min(population, key=lambda x: x.fitness)
            if current_best.fitness < best.fitness:
                best = current_best

            history.append({
                'generation': gen + 1,
                'best_fitness': best.fitness,
                'current_best_fitness': current_best.fitness,
                'best_makespan': best.decoded['makespan'],
                'best_total_qtime_violation': best.decoded['total_qtime_violation']
            })

            if (gen + 1) % 10 == 0:
                print(
                    f"Gen {gen + 1:3d} | "
                    f"Best Fitness={best.fitness:.2f} | "
                    f"Makespan={best.decoded['makespan']:.2f} | "
                    f"Q-viol={best.decoded['total_qtime_violation']:.2f} | "
                )

        history_df = pd.DataFrame(history)
        return best, history_df


def main():
    scheduler = GAScheduler(
        jobs_path='schema/jobs.csv',
        machines_path='schema/machines.csv',
        operations_path='schema/operations.csv',
        op_machine_map_path='schema/operation_machine_map.csv',
        qtime_path='schema/qtime_constraints.csv',
        setup_path='schema/setup_times.csv'
    )

    best, history_df = scheduler.run()


    print('\n========== FINAL RESULT ==========' )
    print(f"Best fitness                : {best.fitness:.2f}")
    print(f"Makespan                    : {best.decoded['makespan']:.2f}")
    print(f"Total qtime violation       : {best.decoded['total_qtime_violation']:.2f}")


if __name__ == '__main__':
    main()
