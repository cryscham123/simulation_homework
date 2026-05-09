import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Any

def create_gantt_chart(logs: List[Any], 
                           max_time:float, 
                           title: str = "반도체 공정 시뮬레이션 간트 차트") -> go.Figure:
    """
    Job 리스트로부터 Gantt Chart를 생성

    Args:
        jobs: Job 인스턴스 리스트
        max_time: 시뮬레이션 최대 시간
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    # 모든 Job의 이벤트 로그 수집
    df_events = pd.DataFrame(logs)

    gantt_data = []

    for id, events in df_events[df_events['resource'] == 'machine'].groupby('id'):
    #for id, events in df_events.groupby('id'):
        for _, event in events.sort_values('start').iterrows():
            if event['start'] == event['finish']:
                continue
            gantt_data.append({
                'Task': id,
                'Start': event['start'],
                'Finish': event['finish'],
                'Resource': event['event'],
                'job_op': event['op_id'] if event['op_id'] is not None else ""
            })

    df_gantt = pd.DataFrame(gantt_data)

    df_gantt['Duration'] = df_gantt['Finish'] - df_gantt['Start']
    color_map = {}
    job_type = []
    for res in df_gantt['Resource'].unique():
        if res == "waiting": color_map[res] = 'rgb(220, 220, 220)'
        elif res == "setup": color_map[res] = 'rgb(0, 200, 83)'
        elif res == "repairing": color_map[res] = 'rgb(255, 65, 54)'
        elif res == "PM": color_map[res] = 'rgb(255, 140, 0)'
        elif res == 'qtime_over': color_map[res] = 'rgb(255, 0, 255)'
        elif "working-" in res:
            val = res.split("-")[1]
            job_type.append(val)
            color_map[res] = f'rgb(0, {200 - len(job_type) * 30}, 255)'
        else:
            color_map[res] = 'rgb(0, 0, 255)'

    target_order = ["qtime_over", "repairing", "PM", "waiting", "setup"] + [f"working-{i}" for i in sorted(job_type)]
    sorted_tasks = sorted(df_gantt['Task'].unique(), key=lambda x: int(x[1:]))

    fig = px.bar(
        df_gantt,
        base="Start",
        x="Duration",
        y="Task",
        color="Resource",
        hover_data=["job_op"],
        labels={"Duration": "Finish"},
        color_discrete_map=color_map,
        orientation='h',
        category_orders={
            "Resource": target_order,
            "Task": sorted_tasks
        },
        title=title,
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Machine ID",
        xaxis=dict(dtick=10),
        height=max(400, len(df_gantt['Task'].unique()) * 50)
    )
    fig.update_traces(width=0.5)
    fig.update_traces(width=0.7, selector=dict(name="qtime_over"))
    return fig
