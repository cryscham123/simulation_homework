import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .chromosome import Chromosome
from .encoder import EncodedData
from .evaluator import Evaluator
from .operators import (
    crossover,
    mutate,
    random_chromosome,
    tournament_select,
)


# worker 프로세스마다 1개씩 갖는 Evaluator 핸들
# multiprocessing은 모듈 top-level 함수만 쓸 수 있어서 클래스 밖에 둠
_worker_evaluator: Optional[Evaluator] = None


def _init_worker(encoded: EncodedData, data: Dict[str, pd.DataFrame], seed: int) -> None:
    """worker 프로세스 시작 시 1회 호출. data를 1번만 받아 worker 메모리에 보관."""
    global _worker_evaluator
    _worker_evaluator = Evaluator(encoded, data, seed=seed)


def _worker_evaluate(chromo: Chromosome) -> Tuple[float, float]:
    """worker에서 chromosome 1개 평가. data는 이미 메모리에 있으니 chromosome만 받음."""
    assert _worker_evaluator is not None
    return _worker_evaluator.evaluate(chromo)


class GA:
    """단일 목적 GA. fitness = makespan + alpha * qtime_violation 최소화."""

    def __init__(self,
                 encoded: EncodedData,
                 data: Dict[str, pd.DataFrame],
                 pop_size: int,
                 n_generations: int,
                 crossover_rate: float,
                 mut_job: float,
                 mut_machine: float,
                 mut_pm: float,
                 tournament_k: int,
                 n_elites: int,
                 alpha: float,
                 seed: int,
                 n_workers: int = 6,     ### 주의!! 각자의 컴퓨터 사양에 따라 사용하는 코어수가 달라짐 확인하고 맞춰서 값을 수정할 것
                                         ### 물리 코어 수를 확인해야함 "작업관리자-CPU" 에서 확인가능
                 verbose: bool = True,
                 verbose_interval: int = 10):
        self.encoded = encoded
        self.evaluator = Evaluator(encoded, data, seed=seed)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mut_job = mut_job
        self.mut_machine = mut_machine
        self.mut_pm = mut_pm
        self.tournament_k = tournament_k
        self.n_elites = n_elites
        self.alpha = alpha
        self.n_workers = n_workers
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        random.seed(seed)

    def fitness_value(self, chromo: Chromosome) -> float:
        """Weighted sum 평가값. 작을수록 좋음."""
        makespan, qtime = chromo.fitness
        return makespan + self.alpha * qtime

    def run(self) -> Tuple[Chromosome, List[Dict[str, Any]]]:
        """GA 실행. (best_chromosome, history) 반환."""
        gen_iter = tqdm(range(1, self.n_generations + 1), desc="GA", unit="gen")

        # Pool을 GA 시작 시 1번 만들고 끝날 때까지 재사용
        # initializer로 data를 worker마다 1번씩만 전송
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker,
            initargs=(self.encoded, self.evaluator.data, self.evaluator.seed),
        ) as pool:

            # 1. 초기 population 생성 + 평가 (병렬)
            population = [random_chromosome(self.encoded) for _ in range(self.pop_size)]
            self._evaluate_batch(pool, population)

            history: List[Dict[str, Any]] = []
            self._record(history, gen=0, population=population)
            if self.verbose:
                self._print(history[-1])

            # 2. 세대 루프
            for gen in gen_iter:
                # 2-1. Elitism: 상위 n_elites개 그대로 보존
                elites = sorted(population, key=self.fitness_value)[:self.n_elites]
                children: List[Chromosome] = list(elites)

                # 2-2. tournament selection → crossover → mutation 으로 자식 채움
                while len(children) < self.pop_size:
                    p1 = tournament_select(population, self.fitness_value, self.tournament_k)
                    p2 = tournament_select(population, self.fitness_value, self.tournament_k)
                    c1, c2 = crossover(p1, p2, self.crossover_rate)
                    c1 = mutate(c1, self.encoded, self.mut_job, self.mut_machine, self.mut_pm)
                    c2 = mutate(c2, self.encoded, self.mut_job, self.mut_machine, self.mut_pm)
                    children.append(c1)
                    if len(children) < self.pop_size:
                        children.append(c2)

                # 2-3. 새 자식만 평가 (병렬, elites는 이미 평가됨)
                self._evaluate_batch(pool, children)

                population = children

                # 2-4. 기록 + 출력
                self._record(history, gen=gen, population=population)
                if self.verbose and (gen % self.verbose_interval == 0 or gen == self.n_generations):
                    self._print(history[-1])

        best = min(population, key=self.fitness_value)
        return best, history

    def _evaluate_batch(self, pool: ProcessPoolExecutor,
                        chromos: List[Chromosome]) -> None:
        """fitness가 비어있는 chromosome들을 worker pool에 나눠서 병렬 평가."""
        to_eval = [c for c in chromos if c.fitness is None]
        if not to_eval:
            return
        results = list(pool.map(_worker_evaluate, to_eval))
        for c, fit in zip(to_eval, results):
            c.fitness = fit

    def _record(self, history: List[Dict[str, Any]], gen: int,
                population: List[Chromosome]) -> None:
        best = min(population, key=self.fitness_value)
        avg_fitness = sum(self.fitness_value(c) for c in population) / len(population)
        history.append({
            'gen': gen,
            'best_makespan': best.fitness[0],
            'best_qtime': best.fitness[1],
            'best_fitness': self.fitness_value(best),
            'avg_fitness': avg_fitness,
        })

    def _print(self, record: Dict[str, Any]) -> None:
        tqdm.write(
            f"[Gen {record['gen']:3d}] "
            f"best_fitness={record['best_fitness']:.2f}  "
            f"makespan={record['best_makespan']:.2f}  "
            f"qtime={record['best_qtime']:.2f}  "
            f"avg={record['avg_fitness']:.2f}"
        )
