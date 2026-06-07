# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Discrete-event simulation of a semiconductor fab scheduling problem, written on top of SimPy. The system simulates jobs flowing through machines grouped by `machine_group`/`op_group`, with setup times, Weibull-distributed failures, preventive maintenance (PM), and q-time constraints. Comparison of dispatching rules (FIFO/SPT/LPT/MIN_QTIME/SPTSSU/COMPOSITE) is the main research output; the COMPOSITE rule itself is tuned by a Shapley + PSO bilevel pipeline.

## Setup & run

```powershell
pip install -r requirements.txt
# Copy .env_sample to .env and fill in values (BASE_DATA_PATH, JOB_RULE, etc.)
```

There is no CLI entrypoint or test suite. Runs happen inside notebooks:
- `result.ipynb` — single simulation run + visualization (Gantt via `utils/visualizer.py` → `gantt_chart.html`).
- `result_test.ipynb` — ad-hoc smoke runs / scratch.
- `result_compare.ipynb` — sweeps multiple `JOB_RULE`s and compares makespan / qtime violation. Also writes `data/normalization_ref.json` (mk_ref/qv_ref from 6 baseline rules × 30 runs).
- `tuning/composite_rule.ipynb` — COMPOSITE rule tuning. Runs Stage 0 (single-term screening) → Stage 1 (Shapley over 2^n−1 subsets with low-budget PSO + elbow on Shapley-ranked cumulative J) → Stage 2 (high-budget PSO on chosen subset) → Stage 3 (held-out validation vs baselines).

Input CSVs live in `data/` (gitignored) and are loaded by `utils/DataLoader.load_all_data()`. All CSV time columns are in **minutes**; output unit conversion is the `TIME_UNIT` env var (`M`/`H`/`D`) applied at `EventLogger.logs` read time only. The tuning pipeline forces `TIME_UNIT=M` internally so J scale is unit-stable.

## Architecture

The simulator is event-driven via SimPy `Store`/`FilterStore` queues. Three actor classes coordinate through two stores owned by `Scheduler`:

- `machine_signal` (Store) — a machine puts itself here whenever it becomes idle (start of sim, after a job completes, after repair/PM finishes). `Stocker.wait_until_machine_ready` consumes from it.
- `machine_events` (Store) — machines request state transitions (REPAIRING/PM) here; `Scheduler.__chk_machine_event` arbitrates priority (REPAIRING preempts PM) and runs `repair()`.
- `job_events` (Store) — jobs notify the scheduler on every state transition; `Scheduler.__chk_job_waiting` re-matches the job to a machine on WAITING and tracks termination.

Dispatch always goes through `Stocker` (`simulation/stocker.py`). A job arriving when a same-group machine is idle is dispatched immediately; otherwise it sits in `Stocker.__resource` (FilterStore). When a machine signals idle, `Stocker.__select_job` picks one job from the same-`op_group` candidates per `JOB_RULE`. **All rules including COMPOSITE share this single selection point** — do not add a separate dispatch path.

`Machine` (`simulation/machine.py`) runs three concurrent processes per machine: `run()` (job execution, low priority on a `PreemptiveResource`), `down()` (Weibull-sampled failure timer), and `PM()` (timer to the configured hazard threshold). Repair takes the resource at priority −1 with `preempt=True` for breakdowns and `preempt=False` for PM. `RepairStatus.SUCCESS_PM` resets the down timer; `FAILED_PM` (PM interrupted by a real breakdown) does not.

`Job` (`simulation/job.py`) tracks operation sequence, q-time (set to `inf` for the first op and for ops with `qtime <= 0`, meaning "no constraint"), and a `__wait_start_time` used by the COMPOSITE rule. `prev_not_completed` is a hand-off flag used to suppress duplicate q-time starts when a job re-enters WAITING after a mid-op machine failure.

### COMPOSITE rule (`simulation/priority.py`)

Linear combination of min-max-normalized terms; **lower π = higher priority**. The rule has two modes — choose by env var, not by code path:

1. **Dynamic mode** (used by the tuning pipeline). Set `COMPOSITE_TERMS='PT,SLACK,...'` and `COMPOSITE_WEIGHTS='w1,w2,...'` (same length). Supports `|S|=1..n` arbitrary subsets — this is what lets Shapley iterate over all 2^n−1 subsets without being capped at 4 slots.
2. **Legacy 4-slot mode** (used by `result.ipynb` and `SAAEvaluator`). When `COMPOSITE_TERMS` is unset, falls back to fixed α/β/γ/δ slots with weights `COMPOSITE_<SLOT>` and per-slot term names `COMPOSITE_<SLOT>_TERM`. Defaults: α=PT, β=SLACK, γ=C_TRANSITION, δ=SETUP.

Registered terms live in `TERM_REGISTRY` (priority.py:187): `PT`, `SLACK`, `SETUP`, `C_TRANSITION`, `COMPLETION_FAST`, `COMPLETION_SLOW`, `WAITING`. Each term registers `(raw_fn, normalizer)`. Sign convention: raw values are flipped so "smaller is more urgent" — the dispatcher picks `argmin π`. Read the term docstrings before changing signs.

Key invariants:
- `SLACK` uses `_min_max_norm_skip_none`: candidates with `remain_qtime=inf` (no q-time constraint) are excluded from the normalization population and assigned `norm=1.0` (always last among risky candidates). Putting them into a normal min-max would flatten the real risk variance — do not "simplify" this.
- `C_TRANSITION` reads `tuning/transition_risk.json` via `load_transition_risk_table()` (written by `SAAEvaluator.__init__`). If the file is absent the c_transition term is silently zero — initialize `SAAEvaluator` once on the project's data before relying on this term.
- Adding a new term = one `_term_xxx` function + one line in `TERM_REGISTRY`. No other code change is needed; the dynamic-mode evaluator and Shapley pipeline pick it up automatically via `available_terms()`.

### Q-time accounting

`Job.__chk_qtime` is started by `Scheduler.__matching_machine` when the job first enters WAITING for an op (`prev_not_completed=False`). It is interrupted at SETUP start. `__waited_time` accumulates across re-waits caused by machine failures during the same op; it is reset only when the op actually completes. Be careful with this when modifying `Machine.run` or `Job.operation_completed`.

### Tuning pipeline (`tuning/`)

- `saa_evaluator.py` — `SAAEvaluator`: legacy 4-slot SAA evaluator. Also owns `_compute_transition_risk()` (deterministic, data-only, no simulation) and writes `tuning/transition_risk.json`. Accepts external `mk_ref`/`qv_ref` (from `data/normalization_ref.json`) to skip its internal reference computation.
- `composite_rule.py` —
  - `DynamicCompositeEvaluator`: like `SAAEvaluator` but uses `COMPOSITE_TERMS`/`COMPOSITE_WEIGHTS` so any `|S|` works. Requires `transition_risk.json` to already exist.
  - `pso_simplex`: minimize-on-simplex PSO. Particles are projected to sum=1 each step; scale-invariant.
  - `value_function(S)`: returns v(S) = min_w J(w;S). |S|=1 skips PSO (w=1.0 by scale-invariance). |S|=0 is defined as 0 (Shapley baseline).
  - `shapley_value`: exact Shapley over all 2^n subsets with a `value_cache` (keys = `frozenset`) so the same v(S) is not recomputed. Sign convention `φ_i = E_S[v(S∪{i}) − v(S)]`, so **φ_i < 0 means term i lowers J (good term)**.
  - `pairwise_interaction`: Grabisch pairwise interaction `I_{ij}` from the same cache. δ<0 = synergy.
  - `cumulative_J_curve`: elbow curve. Ranks terms by ascending φ (most-negative first) and reports v(top-k) for k=1..n. Hits `value_cache` when the prefix subset is already there.
  - `estimate_cost`: pre-flight cost estimate for a full Shapley run.
- `tuning/transition_risk.json`, `tuning/value_cache.json`, `tuning/composite_rule_result.json` — generated artifacts. `value_cache.json` is the hot artifact for the bilevel pipeline; if it looks inconsistent (e.g. a |S|=2 entry's J* matching a |S|=1 entry exactly despite a non-degenerate w*), suspect cache poisoning and re-run Stage 1 rather than patching.

The pipeline is intentionally bilevel: Stage 1 uses **low PSO budget × many subsets** (cheap enough to cover all 2^n subsets, only ordering of φ needs to be correct) and Stage 2 uses **high PSO budget × one subset** (the chosen top-k, where w* must be precise). Don't try to merge the stages.

## Environment variables

All runtime knobs are read from `.env` via `python-dotenv` in the notebooks, then read from `os.getenv` directly inside `simulation/`/`utils/`/`tuning/` modules — there is no central config object. Adding a new knob means: document it in `.env_sample`, read it where used, and (if it affects COMPOSITE) expose it through `SAAEvaluator` or `DynamicCompositeEvaluator`.

The tuning evaluators take ownership of several env vars and overwrite them on each run: `JOB_RULE`, `TIME_UNIT`, `PM_ACTIVE`, `DOWN_ACTIVE`, `PM_RULE`, `COMPOSITE_TERMS`, `COMPOSITE_WEIGHTS`, `COMPOSITE_<SLOT>`, `COMPOSITE_<SLOT>_TERM`. Don't rely on `.env` values for these inside the tuning notebooks.

## Conventions

- Korean comments and docstrings are the norm; keep them when editing.
- The codebase uses name-mangled private attrs (`self.__foo`) extensively. Don't refactor to single-underscore without a reason.
- `program_done()` on `Machine`/`Job` is a manual destructor — Python's `__del__` is unreliable here, so the scheduler explicitly calls it after all jobs complete to close any still-open event log entries.
- Cached artifacts (`value_cache.json`, `transition_risk.json`) are keyed by deterministic seeds and the current data CSVs. If you change data files or the simulation semantics, invalidate them rather than relying on partial overwrites.
