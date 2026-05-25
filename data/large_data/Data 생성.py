import pandas as pd
import numpy as np

# 난수 시드 고정 (재현성을 위해)
np.random.seed(42)

# ==========================================
# 1. machines.csv 생성
# ==========================================
machine_counts = {'G1': 10, 'G2': 1, 'G3': 17, 'G4': 1, 'G5': 1}
machines = []
m_idx = 1
for g, count in machine_counts.items():
    for _ in range(count):
        machines.append({'machine_id': f'M{m_idx}', 'machine_group': g})
        m_idx += 1
df_machines = pd.DataFrame(machines)

# ==========================================
# 2. setup_times.csv 생성
# ==========================================
setup_info = {'G1': 8, 'G2': 8, 'G3': 15, 'G4': 15, 'G5': 8}
job_types = ['P1', 'P2']
setups = []
for g, s_time in setup_info.items():
    for f_t in job_types:
        for t_t in job_types:
            # 같은 타입이면 0, 다른 타입이면 지정된 Setup Time
            t = 0 if f_t == t_t else s_time
            setups.append({
                'machine_group': g, 
                'from_job_type': f_t, 
                'to_job_type': t_t, 
                'setup_time': t
            })
df_setups = pd.DataFrame(setups)

# ==========================================
# 3. jobs.csv 생성
# ==========================================
jobs = []
# P1: 50개 (J1 ~ J50)
for i in range(1, 51):
    jobs.append({'job_id': f'J{i}', 'job_type': 'P1', 'release_time': 0, 'due_date': 10000, 'priority': 1})
# P2: 50개 (J51 ~ J100)
for i in range(51, 101):
    jobs.append({'job_id': f'J{i}', 'job_type': 'P2', 'release_time': 0, 'due_date': 10000, 'priority': 1})
df_jobs = pd.DataFrame(jobs)

# ==========================================
# 4. operations.csv & 5. operation_machine_map.csv 생성
# ==========================================
seq_P1 = ['G3', 'G1', 'G3', 'G1', 'G4', 'G1', 'G3', 'G1', 'G4', 'G1', 'G2', 'G1', 'G2', 'G5']
seq_P2 = ['G3', 'G1', 'G3', 'G1', 'G3', 'G1', 'G4', 'G1', 'G3', 'G1', 'G4', 'G1', 'G2', 'G1']

def get_process_time(p_type, group):
    """지정된 Uniform 분포에 따라 Process Time 생성"""
    if p_type == 'P1':
        if group == 'G1': return np.random.uniform(90, 150)
        elif group == 'G2': return np.random.uniform(35, 55)
        elif group == 'G3': return np.random.uniform(440, 460)
        elif group == 'G4': return np.random.uniform(14.5, 15.5)
        elif group == 'G5': return np.random.uniform(58, 62)
    else: # P2
        if group == 'G1': return np.random.uniform(118, 122)
        elif group == 'G2': return np.random.uniform(39, 41)
        elif group == 'G3': return np.random.uniform(390, 410)
        elif group == 'G4': return np.random.uniform(14.5, 15.5)
        elif group == 'G5': return np.random.uniform(58, 62)

def get_qtime(prev_g, curr_g):
    """이전 공정과 현재 공정을 비교하여 Q-time 반환"""
    if prev_g == 'G1' and curr_g == 'G3': return 600
    if prev_g == 'G1' and curr_g == 'G4': return 120
    if prev_g == 'G4' and curr_g == 'G1': return 1440
    return 0 # 발생하지 않는 경우 0 (또는 빈칸)

ops = []
op_map = []

for job in jobs:
    j_id = job['job_id']
    j_type = job['job_type']
    seq = seq_P1 if j_type == 'P1' else seq_P2
    
    for idx, g in enumerate(seq):
        op_id = f"{j_id}_O{idx+1}"
        op_seq = idx + 1
        
        # Q-time 계산을 위해 이전 그룹 확인
        prev_g = seq[idx-1] if idx > 0 else None
        qtime = get_qtime(prev_g, g)
        
        # operations.csv 데이터
        ops.append({
            'job_id': j_id, 
            'op_id': op_id, 
            'op_seq': op_seq, 
            'op_group': g, 
            'qtime': qtime
        })
        
        # operation_machine_map.csv 데이터
        # 현재 그룹에 속한 모든 Machine에 대해 Process Time 생성
        group_machines = df_machines[df_machines['machine_group'] == g]['machine_id'].tolist()
        for m in group_machines:
            pt = round(get_process_time(j_type, g), 2) # 소수점 둘째 자리까지 반올림
            op_map.append({
                'op_id': op_id, 
                'machine_id': m, 
                'process_time': pt
            })

df_ops = pd.DataFrame(ops)
df_op_map = pd.DataFrame(op_map)

# ==========================================
# 6. machine_failure.csv 생성
# ==========================================
# 고장 관련 요구사항은 별도로 없으므로, 생성된 30대의 장비에 기존 Base Data를 일괄 적용합니다.
failures = []
for m in df_machines['machine_id']:
    failures.append({
        'machine_id': m,
        'base_hazard': 0.005,
        'hazard_increase_rate': 0.0005,
        'repair_time': 10,
        'pm_duration': 5
    })
df_failures = pd.DataFrame(failures)

# ==========================================
# CSV 파일로 저장
# ==========================================
df_machines.to_csv('machines.csv', index=False)
df_setups.to_csv('setup_times.csv', index=False)
df_jobs.to_csv('jobs.csv', index=False)
df_ops.to_csv('operations.csv', index=False)
df_op_map.to_csv('operation_machine_map.csv', index=False)
df_failures.to_csv('machine_failure.csv', index=False)