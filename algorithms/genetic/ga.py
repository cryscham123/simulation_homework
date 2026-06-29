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
    """단일 목적 GA. fitness = makespan/mk_ref + alpha * AQV/qv_ref 최소화 (정규화)."""

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
                 mk_ref: float,
                 qv_ref: float,
                 seed: int,
                 stagnation_patience: int = 10,
                 mutation_boost: float = 8.0,
                 immigrant_ratio: float = 0.2,
                 boost_duration: int = 5,
                 boost_cooldown: int = 10,
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
        # ref를 GA 내부 단위(시간 + job당 평균)로 변환해서 보관
        self.mk_ref = mk_ref / 60.0                          # 분 → 시간
        self.qv_ref = qv_ref / len(data['jobs']) / 60.0      # total 분 → job당 평균 시간
        self.stagnation_patience = stagnation_patience
        self.mutation_boost = mutation_boost
        self.immigrant_ratio = immigrant_ratio
        self.boost_duration = boost_duration
        self.boost_cooldown = boost_cooldown
        self.n_workers = n_workers
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        random.seed(seed)

    def fitness_value(self, chromo: Chromosome) -> float:
        """정규화 weighted sum. 작을수록 좋음."""
        makespan, qtime = chromo.fitness
        return makespan / self.mk_ref + self.alpha * (qtime / self.qv_ref)

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
            self._record(history, gen=0, population=population, boost_active=False)
            if self.verbose:
                self._print(history[-1])

            # 정체 추적 상태
            best_so_far = history[-1]['best_fitness']
            no_improve = 0
            active_left = 0    # boost ON으로 남은 세대 수
            cooldown_left = 0  # boost 강제 OFF로 남은 세대 수
            n_immigrants = int(self.immigrant_ratio * self.pop_size)

            # 2. 세대 루프
            for gen in gen_iter:
                # 2-0. boost 사이클 진입 여부 판단 (idle 상태일 때만 트리거)
                if active_left == 0 and cooldown_left == 0 and no_improve >= self.stagnation_patience:
                    active_left = self.boost_duration

                boost_active = active_left > 0
                if boost_active and n_immigrants > 0:
                    # 하위 n_immigrants 명을 새 random으로 교체 (elite는 위쪽이라 영향 없음)
                    population = sorted(population, key=self.fitness_value)
                    for i in range(self.pop_size - n_immigrants, self.pop_size):
                        population[i] = random_chromosome(self.encoded)
                    self._evaluate_batch(pool, population)

                mj, mm, mp = self._current_mutation_rates(boost_active)

                # 2-1. Elitism: 상위 n_elites개 그대로 보존
                elites = sorted(population, key=self.fitness_value)[:self.n_elites]
                children: List[Chromosome] = list(elites)

                # 2-2. tournament selection → crossover → mutation 으로 자식 채움
                while len(children) < self.pop_size:
                    p1 = tournament_select(population, self.fitness_value, self.tournament_k)
                    p2 = tournament_select(population, self.fitness_value, self.tournament_k)
                    c1, c2 = crossover(p1, p2, self.crossover_rate)
                    c1 = mutate(c1, self.encoded, mj, mm, mp)
                    c2 = mutate(c2, self.encoded, mj, mm, mp)
                    children.append(c1)
                    if len(children) < self.pop_size:
                        children.append(c2)

                # 2-3. 새 자식만 평가 (병렬, elites는 이미 평가됨)
                self._evaluate_batch(pool, children)

                population = children

                # 2-4. 기록 + 출력
                self._record(history, gen=gen, population=population, boost_active=boost_active)
                if self.verbose and (gen % self.verbose_interval == 0 or gen == self.n_generations):
                    self._print(history[-1])

                # 2-5. 정체 카운터 + boost 사이클 갱신
                current_best = history[-1]['best_fitness']
                if current_best < best_so_far:
                    # 개선 발생 → 카운터/사이클 즉시 리셋, exploit 단계로
                    best_so_far = current_best
                    no_improve = 0
                    active_left = 0
                    cooldown_left = 0
                else:
                    no_improve += 1
                    # active → 끝나면 cooldown으로 전환, cooldown → 끝나면 idle
                    if active_left > 0:
                        active_left -= 1
                        if active_left == 0:
                            cooldown_left = self.boost_cooldown
                    elif cooldown_left > 0:
                        cooldown_left -= 1

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

    def _current_mutation_rates(self, boost_active: bool) -> Tuple[float, float, float]:
        """boost 모드면 mutation 3개에 배율 적용. 1.0 clamp."""
        b = self.mutation_boost if boost_active else 1.0
        return (
            min(1.0, self.mut_job * b),
            min(1.0, self.mut_machine * b),
            min(1.0, self.mut_pm * b),
        )

    def _record(self, history: List[Dict[str, Any]], gen: int,
                population: List[Chromosome], boost_active: bool) -> None:
        best = min(population, key=self.fitness_value)
        avg_fitness = sum(self.fitness_value(c) for c in population) / len(population)
        history.append({
            'gen': gen,
            'best_makespan': best.fitness[0],
            'best_qtime': best.fitness[1],
            'best_fitness': self.fitness_value(best),
            'avg_fitness': avg_fitness,
            'boost_active': boost_active,
        })

    def _print(self, record: Dict[str, Any]) -> None:
        boost_tag = "  [BOOST]" if record.get('boost_active') else ""
        tqdm.write(
            f"[Gen {record['gen']:3d}] "
            f"best_fitness={record['best_fitness']:.2f}  "
            f"makespan={record['best_makespan']:.2f}  "
            f"qtime={record['best_qtime']:.2f}  "
            f"avg={record['avg_fitness']:.2f}"
            f"{boost_tag}"
        )
