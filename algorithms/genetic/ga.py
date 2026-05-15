import random
from typing import Any, Dict, List, Tuple

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

        # 1. 초기 population 생성 + 평가
        population = [random_chromosome(self.encoded) for _ in range(self.pop_size)]
        for c in population:
            c.fitness = self.evaluator.evaluate(c)

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

            # 2-3. 새 자식만 평가 (elites는 이미 평가됨)
            for c in children:
                if c.fitness is None:
                    c.fitness = self.evaluator.evaluate(c)

            population = children

            # 2-4. 기록 + 출력
            self._record(history, gen=gen, population=population)
            if self.verbose and (gen % self.verbose_interval == 0 or gen == self.n_generations):
                self._print(history[-1])

        best = min(population, key=self.fitness_value)
        return best, history

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
