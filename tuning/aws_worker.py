"""
AWS 분산 실행용 워커 스크립트.

각 AWS 인스턴스는 자신의 .env 파일에 COMPOSITE_TERMS를 설정하고 이 스크립트를 실행한다.
v(S) = min_w J(w; S) 를 계산한 뒤 결과를 JSON으로 저장한다.

.env 예시 (인스턴스마다 COMPOSITE_TERMS 만 다름):
    COMPOSITE_TERMS=SLACK,SETUP,C_TRANSITION
    BASE_DATA_PATH=data
    PM_ACTIVE=FALSE
    DOWN_ACTIVE=TRUE

실행:
    python tuning/aws_worker.py
    python tuning/aws_worker.py --n_runs 10 --swarm_size 8 --n_iter 15 --n_workers 8

결과 파일:
    tuning/aws_results/subset_<항목_정렬_나열>.json
"""
import os
import sys
import json
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

from utils import DataLoader
from tuning.saa_evaluator import SAAEvaluator, TRANSITION_RISK_CACHE_PATH
from tuning.composite_rule import (
    load_normalization_ref,
    DynamicCompositeEvaluator,
    value_function,
)


def main():
    parser = argparse.ArgumentParser(description='AWS 워커: 단일 부분집합 v(S) 계산')
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'tuning', 'aws_results'),
                        help='결과 JSON 저장 디렉토리')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='SAA replication 수 (default: 10)')
    parser.add_argument('--swarm_size', type=int, default=8,
                        help='PSO swarm 크기 (default: 8)')
    parser.add_argument('--n_iter', type=int, default=15,
                        help='PSO iteration 수 (default: 15)')
    parser.add_argument('--pso_seed', type=int, default=0,
                        help='PSO 탐색 seed (default: 0)')
    parser.add_argument('--base_seed', type=int, default=0,
                        help='시뮬레이션 CRN base seed (default: 0)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='n_runs 병렬화 워커 수 (default: 1, >1이면 multiprocessing 사용)')
    args = parser.parse_args()

    # .env에서 COMPOSITE_TERMS 읽기
    terms_str = os.environ.get('COMPOSITE_TERMS', '').strip()
    if not terms_str:
        print("ERROR: .env에 COMPOSITE_TERMS가 설정되지 않았습니다.")
        print("예시: COMPOSITE_TERMS=SLACK,SETUP,C_TRANSITION")
        sys.exit(1)

    terms = sorted([t.strip().upper() for t in terms_str.split(',') if t.strip()])
    print(f"[worker] 부분집합 S = {terms}")
    print(f"[worker] PSO: swarm={args.swarm_size}, n_iter={args.n_iter}, "
          f"n_runs={args.n_runs}, n_workers={args.n_workers}")

    # 데이터 로드
    base_data_path = os.environ.get('BASE_DATA_PATH', 'data').strip()
    if not os.path.isabs(base_data_path):
        base_data_path = os.path.join(PROJECT_ROOT, base_data_path)
    data = DataLoader(base_data_path).load_all_data()
    print(f"[worker] 데이터 로드: jobs={len(data['jobs'])}, machines={len(data['machines'])}")

    # 정규화 기준 로드
    norm_ref_path = os.path.join(PROJECT_ROOT, 'data', 'normalization_ref.json')
    mk_ref, qv_ref_aqt, _ = load_normalization_ref(norm_ref_path)
    print(f"[worker] mk_ref={mk_ref:.3f}, qv_ref_aqt={qv_ref_aqt:.3f}")

    # transition_risk 캐시 확인 (없으면 생성)
    if not os.path.exists(TRANSITION_RISK_CACHE_PATH):
        print("[worker] transition_risk 캐시 없음 → SAAEvaluator로 생성 중...")
        SAAEvaluator(data, obj_weights=(1.0, 1.0), base_seed=0,
                     down_active=True, pm_active=False,
                     mk_ref=mk_ref, qv_ref=qv_ref_aqt)

    # 평가기 초기화
    evaluator = DynamicCompositeEvaluator(
        data,
        mk_ref=mk_ref,
        qv_ref_aqt=qv_ref_aqt,
        base_seed=args.base_seed,
        down_active=True,
        pm_active=False,
    )

    pso_kwargs = dict(swarm_size=args.swarm_size, n_iter=args.n_iter, seed=args.pso_seed)

    # v(S) 계산
    t0 = time.time()
    result = value_function(evaluator, terms, args.n_runs, pso_kwargs, n_workers=args.n_workers)
    elapsed = time.time() - t0

    print(f"[worker] 완료: J*={result['J*']:.5f}  w*={dict(zip(terms, [round(float(x), 4) for x in result['w*']]))}  elapsed={elapsed:.1f}s")

    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    key = '_'.join(terms)
    output_path = os.path.join(args.output_dir, f"subset_{key}.json")

    payload = {
        'terms': terms,
        'J_star': float(result['J*']),
        'w_star': [float(x) for x in result['w*']],
        'elapsed_sec': elapsed,
        'config': {
            'n_runs': args.n_runs,
            'pso_kwargs': pso_kwargs,
            'base_seed': args.base_seed,
            'n_workers': args.n_workers,
        },
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[worker] 저장 완료: {output_path}")


if __name__ == '__main__':
    main()
