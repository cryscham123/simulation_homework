"""
Shapley part 병합 스크립트.

AWS 인스턴스들이 Google Drive 로 업로드한 part_*.json (Shapley_cal.ipynb 산출물)을
한 디렉토리에 내려받은 뒤, composite_rule.ipynb 가 바로 쓰는 value_cache.json 형식으로
병합한다.

워크플로:
    1. (repo #1 codespace) gen_shapley_envs.py 로 envs/.env.N 생성 → make simulation
    2. 각 EC2 인스턴스가 part_<TAG>.json 을 Google Drive 업로드
    3. Drive 에서 part_*.json 들을 로컬 폴더(기본: tuning/shapley_results/)로 내려받기
    4. python tuning/shapley_merge.py
       → tuning/value_cache.json 갱신 (64개 부분집합 완비 시 완성 메시지)
    5. composite_rule.ipynb 재실행 → load_value_cache() 가 캐시 hit → Shapley 즉시 산출

사용:
    python tuning/shapley_merge.py
    python tuning/shapley_merge.py --input_dir tuning/shapley_results \
        --output tuning/value_cache.json \
        --candidate_terms PT,SLACK,SETUP,C_TRANSITION,COMPLETION_FAST,WAITING
"""
import os
import sys
import json
import glob
import argparse
from itertools import combinations

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _key(terms):
    return tuple(sorted(t.upper() for t in terms))


def main():
    p = argparse.ArgumentParser(description='Shapley part_*.json → value_cache.json 병합')
    p.add_argument('--input_dir',
                   default=os.path.join(PROJECT_ROOT, 'tuning', 'shapley_results'),
                   help='part_*.json 들이 있는 디렉토리')
    p.add_argument('--output',
                   default=os.path.join(PROJECT_ROOT, 'tuning', 'value_cache.json'),
                   help='출력 value_cache.json 경로')
    p.add_argument('--candidate_terms',
                   default='PT,SLACK,SETUP,C_TRANSITION,COMPLETION_FAST,WAITING',
                   help='전체 후보 항 (쉼표 구분) — 완성도 검사 기준')
    p.add_argument('--keep_existing', action='store_true',
                   help='기존 output 의 entry 도 병합(기본은 part 들로만 새로 구성)')
    args = p.parse_args()

    candidate_terms = [t.strip().upper() for t in args.candidate_terms.split(',') if t.strip()]
    n_terms = len(candidate_terms)
    expected_total = 2 ** n_terms  # 공집합 포함
    print(f'후보 항 ({n_terms}개): {candidate_terms}')
    print(f'기대 부분집합 수(공집합 포함): {expected_total}')

    merged = {}   # key(tuple sorted) -> entry dict
    meta_src = {}

    # (옵션) 기존 value_cache.json 누적
    if args.keep_existing and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            old = json.load(f)
        meta_src = dict(old.get('meta', {}))
        for e in old.get('entries', []):
            merged[_key(e['terms'])] = {
                'terms': sorted(t.upper() for t in e['terms']),
                'J*': float(e['J*']),
                'w*': [float(x) for x in e.get('w*', [])],
            }
        print(f"기존 value_cache.json 에서 {len(merged)}개 로드")

    # part_*.json 읽기
    files = sorted(glob.glob(os.path.join(args.input_dir, 'part_*.json')))
    if not files:
        print(f'[경고] {args.input_dir} 에 part_*.json 이 없습니다.')
    print(f'입력 part 파일 {len(files)}개: {args.input_dir}')

    new_count, dup_count = 0, 0
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        # 첫 part 의 meta 를 대표값으로 채택 (없으면 유지)
        for k in ('n_runs', 'pso_kwargs', 'base_seed', 'down_active',
                  'pm_active', 'mk_ref', 'qv_ref_aqt'):
            if k not in meta_src and k in d.get('meta', {}):
                meta_src[k] = d['meta'][k]
        for e in d.get('entries', []):
            k = _key(e['terms'])
            entry = {
                'terms': sorted(t.upper() for t in e['terms']),
                'J*': float(e['J*']),
                'w*': [float(x) for x in e.get('w*', [])],
            }
            if k in merged:
                dup_count += 1
            else:
                new_count += 1
            merged[k] = entry  # 최신 우선
    print(f'새 entry {new_count}개, 중복(덮어씀) {dup_count}개, 누적 {len(merged)}개')

    # 완성도 검사 (공집합 포함 2^n)
    missing = []
    for k in range(0, n_terms + 1):
        for combo in combinations(candidate_terms, k):
            if _key(combo) not in merged:
                missing.append(list(combo))
    if missing:
        print(f'\n[경고] 누락 부분집합 {len(missing)}/{expected_total}개:')
        for m in missing[:15]:
            print('   ', m if m else '∅ (EMPTY/FIFO)')
        if len(missing) > 15:
            print(f'   ... 외 {len(missing) - 15}개')
    else:
        print(f'\n[완성] 전체 {expected_total}개 부분집합(공집합 포함) 모두 확보!')

    # value_cache.json 형식으로 저장 (composite_rule.ipynb load_value_cache 호환)
    payload = {
        'meta': {
            'candidate_terms': candidate_terms,
            'n_runs': meta_src.get('n_runs'),
            'pso_kwargs': meta_src.get('pso_kwargs'),
            'mk_ref': meta_src.get('mk_ref'),
            'qv_ref_aqt': meta_src.get('qv_ref_aqt'),
            'base_seed': meta_src.get('base_seed'),
            'down_active': meta_src.get('down_active'),
            'pm_active': meta_src.get('pm_active'),
            'merged_from_shapley_parts': True,
            'n_part_files': len(files),
            'total_entries': len(merged),
        },
        'entries': list(merged.values()),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'\n저장 완료: {args.output}  ({len(merged)}개 entry)')
    if not missing:
        print('다음: composite_rule.ipynb 재실행 → 캐시 hit 으로 Shapley 즉시 산출')


if __name__ == '__main__':
    main()
