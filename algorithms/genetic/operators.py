import random
from typing import Callable, List, Tuple

from .chromosome import Chromosome
from .encoder import EncodedData


def random_chromosome(encoded: EncodedData) -> Chromosome:
    """무작위 초기 염색체 1개 생성."""
    n_jobs = len(encoded.job_index_table)
    n_machines = len(encoded.machine_index_table)
    n_levels = len(encoded.pm_levels)

    # job_seq: 0~J-1을 무작위로 섞은 순열 (각 job이 정확히 한 번씩 등장)
    job_seq = random.sample(range(n_jobs), n_jobs)

    # machine: 각 op별로 feasible 머신 중 하나를 무작위 선택
    machine = [random.randrange(len(feasible)) for feasible in encoded.feasible_machine_table]

    # pm: 각 머신별로 PM 레벨 중 하나를 무작위 선택
    pm = [random.randrange(n_levels) for _ in range(n_machines)]

    return Chromosome(job_seq=job_seq, machine=machine, pm=pm)


### Selection
def tournament_select(population: List[Chromosome],
                      fitness_fn: Callable[[Chromosome], float],
                      k: int = 3) -> Chromosome:
    """k명 무작위 추출 후 fitness_fn 기준 최소값(=가장 좋음) 선발."""
    candidates = random.sample(population, k)
    return min(candidates, key=fitness_fn)


#### Crossover

def crossover(parent1: Chromosome,
              parent2: Chromosome,
              rate: float = 0.8) -> Tuple[Chromosome, Chromosome]:
    """염색체 부분별로 다른 crossover. rate 확률로 교차, 아니면 부모 그대로 복사."""
    if random.random() >= rate:
        return _copy(parent1), _copy(parent2)

    job_c1, job_c2 = _ox_crossover(parent1.job_seq, parent2.job_seq)
    mac_c1, mac_c2 = _uniform_crossover(parent1.machine, parent2.machine)
    pm_c1, pm_c2 = _uniform_crossover(parent1.pm, parent2.pm)

    return (
        Chromosome(job_seq=job_c1, machine=mac_c1, pm=pm_c1),
        Chromosome(job_seq=job_c2, machine=mac_c2, pm=pm_c2),
    )


def _copy(chromo: Chromosome) -> Chromosome:
    return Chromosome(
        job_seq=list(chromo.job_seq),
        machine=list(chromo.machine),
        pm=list(chromo.pm),
    )


def _ox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover. 두 순열 부모 → 두 순열 자식."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))   # segment 범위 [a, b]

    def make_child(seg_parent: List[int], fill_parent: List[int]) -> List[int]:
        child = [None] * n
        child[a:b + 1] = seg_parent[a:b + 1]
        used = set(child[a:b + 1])
        fill = (g for g in fill_parent if g not in used)
        for i in range(n):
            if child[i] is None:
                child[i] = next(fill)
        return child

    return make_child(p1, p2), make_child(p2, p1)


def _uniform_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Uniform Crossover. 각 위치마다 50% 확률로 부모 교환."""
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):
        if random.random() < 0.5:
            c1.append(g1)
            c2.append(g2)
        else:
            c1.append(g2)
            c2.append(g1)
    return c1, c2


#### Mutation

def mutate(chromo: Chromosome,
           encoded: EncodedData,
           rate: float = 0.1) -> Chromosome:
    """각 부분 독립적으로 rate 확률로 변이. 새 Chromosome 반환 (원본 보존)."""
    new_job = list(chromo.job_seq)
    new_mac = list(chromo.machine)
    new_pm = list(chromo.pm)

    if random.random() < rate:
        _swap_mutation(new_job)
    if random.random() < rate:
        _random_reset_machine(new_mac, encoded.feasible_machine_table)
    if random.random() < rate:
        _random_reset(new_pm, len(encoded.pm_levels))

    return Chromosome(job_seq=new_job, machine=new_mac, pm=new_pm)


def _swap_mutation(seq: List[int]) -> None:
    """두 위치를 무작위로 swap. 순열 보존."""
    if len(seq) < 2:
        return
    i, j = random.sample(range(len(seq)), 2)
    seq[i], seq[j] = seq[j], seq[i]


def _random_reset_machine(genes: List[int],
                           feasible_table: List[List[str]]) -> None:
    """machine 유전자 한 위치를 다른 feasible 인덱스로 재설정."""
    i = random.randrange(len(genes))
    feasible_count = len(feasible_table[i])
    if feasible_count < 2:
        return
    new_val = random.randrange(feasible_count)
    while new_val == genes[i]:
        new_val = random.randrange(feasible_count)
    genes[i] = new_val


def _random_reset(genes: List[int], n_values: int) -> None:
    """일반 정수 유전자 한 위치를 다른 값으로 재설정."""
    if n_values < 2:
        return
    i = random.randrange(len(genes))
    new_val = random.randrange(n_values)
    while new_val == genes[i]:
        new_val = random.randrange(n_values)
    genes[i] = new_val
