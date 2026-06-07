"""
AWS 결과 병합 스크립트.

tuning/aws_results/ 의 subset_*.json 파일들을 읽어
composite_rule.ipynb 가 바로 사용할 수 있는 value_cache.json 형식으로 병합한다.

실행:
    python tuning/aws_merge.py
    python tuning/aws_merge.py --input_dir tuning/aws_results --output tuning/value_cache.json

병합 후 composite_rule.ipynb 에서:
    value_cache = load_value_cache()   # 64개 전부 로드
    phi, value_cache, meta = shapley_value(...)  # 캐시 hit → 즉시 완료 (0.0s)
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description='AWS 결과 병합: subset JSON → value_cache.json')
    parser.add_argument('--input_dir',
                        default=os.path.join(PROJECT_ROOT, 'tuning', 'aws_results'),
                        help='subset_*.json 파일이 있는 디렉토리')
    parser.add_argument('--output',
                        default=os.path.join(PROJECT_ROOT, 'tuning', 'value_cache.json'),
                        help='출력 value_cache.json 경로')
    parser.add_argument('--candidate_terms',
                        default='PT,SLACK,SETUP,C_TRANSITION,COMPLETION_FAST,WAITING',
                        help='전체 후보 항 (쉼표 구분)')
    parser.add_argument('--base_seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--merge_empty_set', action='store_true',
                        help='빈 집합 v(∅) 를 FIFO 결과로 추가 (별도 계산 필요 시)')
    args = parser.parse_args()

    candidate_terms = [t.strip().upper() for t in args.candidate_terms.split(',')]
    n_terms = len(candidate_terms)
    expected_nonempty = 2 ** n_terms - 1
    print(f"후보 항 ({n_terms}개): {candidate_terms}")
    print(f"예상 비어 있지 않은 부분집합 수: {expected_nonempty}")

    # 기존 value_cache.json 로드 (있으면 누적 병합)
    existing_entries: dict = {}
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        for e in existing.get('entries', []):
            key = str(sorted(e['terms']))
            existing_entries[key] = e
        print(f"기존 value_cache.json에서 {len(existing_entries)}개 로드")

    # aws_results 의 subset_*.json 읽기
    pattern = os.path.join(args.input_dir, 'subset_*.json')
    files = sorted(glob.glob(pattern))
    print(f"입력 파일 {len(files)}개 발견: {args.input_dir}")

    new_count = 0
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            d = json.load(f)
        terms = sorted([t.upper() for t in d['terms']])
        key = str(terms)
        entry = {
            'terms': terms,
            'J*': float(d['J_star']),
            'w*': [float(x) for x in d['w_star']],
        }
        if key not in existing_entries:
            new_count += 1
        existing_entries[key] = entry  # 덮어쓰기 (최신 결과 우선)

    print(f"새로 추가된 항목: {new_count}개 / 전체: {len(existing_entries)}개")

    # 완성도 체크
    missing = []
    from itertools import combinations
    for k in range(1, n_terms + 1):
        for combo in combinations(candidate_terms, k):
            key = str(sorted(combo))
            if key not in existing_entries:
                missing.append(list(sorted(combo)))

    if missing:
        print(f"\n[경고] 아직 계산되지 않은 부분집합 {len(missing)}개:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... 외 {len(missing)-10}개")
    else:
        print(f"\n[완성] 모든 {expected_nonempty}개 비어 있지 않은 부분집합 계산 완료!")

    # 빈 집합 (FIFO baseline) 처리
    empty_key = str([])
    if empty_key not in existing_entries:
        # 별도 FIFO 계산 결과가 없으면 경고만
        print("\n[주의] v(∅) (FIFO baseline) 이 없습니다. "
              "composite_rule.ipynb 실행 시 shapley_value()가 자동 계산합니다.")
    else:
        print(f"v(∅) = {existing_entries[empty_key]['J*']:.5f}  (FIFO baseline)")

    # value_cache.json 형식으로 저장
    # 빈 집합 entry는 terms=[] 로 저장
    all_entries = list(existing_entries.values())

    # PSO kwargs는 첫 번째 항목에서 추출 (있으면)
    sample_config = {}
    for fpath in files[:1]:
        try:
            with open(fpath) as f:
                d = json.load(f)
            sample_config = d.get('config', {})
        except Exception:
            pass

    payload = {
        'meta': {
            'candidate_terms': candidate_terms,
            'n_runs': sample_config.get('n_runs', args.n_runs),
            'pso_kwargs': sample_config.get('pso_kwargs', {}),
            'base_seed': sample_config.get('base_seed', args.base_seed),
            'merged_from_aws': True,
            'total_entries': len(all_entries),
        },
        'entries': all_entries,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {args.output}  ({len(all_entries)}개 항목)")
    print("\n다음 단계:")
    print("  composite_rule.ipynb 를 열고 셀 재실행 →")
    print("  load_value_cache() 로 64개 로드 → shapley_value() 0.0s 완료")


if __name__ == '__main__':
    main()
