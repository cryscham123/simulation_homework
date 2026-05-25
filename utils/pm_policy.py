"""
PM (Preventive Maintenance) Policy Utilities

This module provides functions to calculate and build PM policies based on Weibull distribution.
"""

import pandas as pd
import numpy as np
from math import gamma, log
from typing import Dict, Tuple, Optional

# Module-level PM interval store for simulation
PM_INTERVALS: Dict[str, float] = {}  # machine_id -> pm_interval (time units)



def calculate_weibull_mttf(shape: float, scale: float) -> float:
    """
    Calculate Mean Time To Failure (MTTF) using Weibull distribution.
    
    MTTF = scale × Gamma(1 + 1/shape)
    
    Args:
        shape: Weibull shape parameter (k)
        scale: Weibull scale parameter (λ)
    
    Returns:
        float: MTTF value
    """
    if shape <= 0 or scale <= 0:
        return float('inf')
    
    return scale * gamma(1 + 1 / shape)


def weibull_time_at_failure_prob(shape: float, scale: float, prob: float) -> float:
    """
    Calculate time at which failure probability reaches a given level.
    
    Formula: t_p = scale × (-ln(1 - p))^(1/shape)
    
    Args:
        shape: Weibull shape parameter (k)
        scale: Weibull scale parameter (λ)
        prob: Failure probability (0 < prob < 1)
    
    Returns:
        float: Time at which failure probability = prob
    """
    if shape <= 0 or scale <= 0:
        return float('inf')
    if prob <= 0 or prob >= 1:
        return float('inf')
    
    return scale * ((-log(1.0 - prob)) ** (1.0 / shape))


def build_mttf_pm_policy(
    machines_df: pd.DataFrame,
    machine_failure_df: pd.DataFrame,
    group_risk_type: Dict[str, str],
    risk_coefficient: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Build PM policy based on Weibull MTTF for each machine group.
    
    This function:
    1. Merges machine and failure data
    2. Calculates MTTF for each machine
    3. Assigns risk type to each machine group
    4. Calculates PM interval = coefficient × MTTF
    
    Args:
        machines_df: DataFrame with columns: machine_id, machine_group, ...
        machine_failure_df: DataFrame with columns: machine_id, shape parameter, 
                           scale parameter, repair_time, pm_duration
        group_risk_type: Dict mapping machine_group -> risk_type (high/medium/low)
                        Example: {"G1": "medium", "G2": "high", ...}
        risk_coefficient: Dict mapping risk_type -> coefficient
                         Example: {"high": 0.3, "medium": 0.5, "low": 0.7}
    
    Returns:
        Tuple of (machine_pm_policy, group_pm_policy, pm_interval_dict):
        - machine_pm_policy: DataFrame with machine-level PM info
        - group_pm_policy: DataFrame with group-level PM summary
        - pm_interval_dict: Dict mapping machine_id -> pm_interval for simulation
    """
    # Merge machines and failure data
    merged = machines_df[['machine_id', 'machine_group']].merge(
        machine_failure_df,
        on='machine_id',
        how='left'
    )
    
    # Calculate MTTF for each machine
    merged['mttf'] = merged.apply(
        lambda row: calculate_weibull_mttf(
            row['shape parameter'],
            row['scale parameter']
        ),
        axis=1
    )
    
    # Add risk type and PM coefficient
    merged['risk_type'] = merged['machine_group'].map(group_risk_type)
    merged['pm_coefficient'] = merged['risk_type'].map(risk_coefficient)
    
    # Calculate PM interval
    merged['pm_interval'] = merged['mttf'] * merged['pm_coefficient']
    
    # Create machine PM policy DataFrame
    machine_pm_policy = merged[[
        'machine_id',
        'machine_group',
        'shape parameter',
        'scale parameter',
        'mttf',
        'risk_type',
        'pm_coefficient',
        'pm_interval',
        'repair_time',
        'pm_duration'
    ]].copy()
    
    # Create group PM policy DataFrame
    group_pm_policy = merged.groupby('machine_group').agg({
        'machine_id': 'count',
        'shape parameter': 'mean',
        'scale parameter': 'mean',
        'mttf': 'mean',
        'repair_time': 'mean',
        'pm_duration': 'mean',
        'risk_type': 'first',  # Assuming all machines in group have same risk type
        'pm_coefficient': 'first',
        'pm_interval': 'mean'
    }).reset_index()
    
    group_pm_policy.rename(columns={'machine_id': 'machine_count'}, inplace=True)
    group_pm_policy.rename(columns={
        'shape parameter': 'avg_shape_parameter',
        'scale parameter': 'avg_scale_parameter',
        'mttf': 'avg_mttf',
        'repair_time': 'avg_repair_time',
        'pm_duration': 'avg_pm_duration',
        'pm_interval': 'avg_pm_interval'
    }, inplace=True)
    
    # Reorder columns
    group_pm_policy = group_pm_policy[[
        'machine_group',
        'machine_count',
        'avg_shape_parameter',
        'avg_scale_parameter',
        'avg_mttf',
        'avg_repair_time',
        'avg_pm_duration',
        'risk_type',
        'pm_coefficient',
        'avg_pm_interval'
    ]]
    
    # Create pm_interval_dict for simulation
    pm_interval_dict = dict(zip(
        machine_pm_policy['machine_id'],
        machine_pm_policy['pm_interval']
    ))
    
    return machine_pm_policy, group_pm_policy, pm_interval_dict


def summarize_machine_group_info(
    machines_df: pd.DataFrame,
    operation_machine_map_df: pd.DataFrame,
    operations_df: pd.DataFrame,
    qtime_constraints_df: pd.DataFrame = None,
    qtime_sensitive_threshold: int = 120
) -> pd.DataFrame:
    """
    Summarize machine group characteristics including workload and q-time sensitivity.
    
    Args:
        machines_df: DataFrame with machine information
        operation_machine_map_df: DataFrame mapping operations to machines
        operations_df: DataFrame with operation details
        qtime_constraints_df: Optional DataFrame with q-time constraints
        qtime_sensitive_threshold: Threshold for q-time sensitive (default: 120 minutes)
    
    Returns:
        DataFrame: Group summary with columns:
        - machine_group
        - machine_count
        - total_available_operations
        - avg_processing_time
        - total_processing_time
        - qtime_sensitive_ratio (if qtime_constraints_df provided)
    """
    # Count machines per group
    group_counts = machines_df.groupby('machine_group')['machine_id'].count()
    
    # Count operations per group
    group_ops = operation_machine_map_df.merge(
        operations_df[['op_id', 'process_time']],
        on='op_id',
        how='left'
    ).groupby('machine_group').agg({
        'op_id': 'count',
        'process_time': ['sum', 'mean']
    }).reset_index()
    
    group_ops.columns = ['machine_group', 'total_available_operations', 
                         'total_processing_time', 'avg_processing_time']
    
    # Add machine count
    result = group_ops.copy()
    result['machine_count'] = result['machine_group'].map(group_counts)
    
    # Add q-time sensitivity if constraint data provided
    if qtime_constraints_df is not None:
        qtime_sensitive = qtime_constraints_df[
            (qtime_constraints_df['qtime'] > 0) & 
            (qtime_constraints_df['qtime'] <= qtime_sensitive_threshold)
        ].groupby('op_id').size()
        
        qtime_ops = operation_machine_map_df.merge(
            qtime_constraints_df[['op_id', 'qtime']],
            on='op_id',
            how='left'
        )
        
        qtime_ops['is_sensitive'] = (
            (qtime_ops['qtime'] > 0) & 
            (qtime_ops['qtime'] <= qtime_sensitive_threshold)
        ).astype(int)
        
        qtime_summary = qtime_ops.groupby('machine_group').agg({
            'op_id': 'count',
            'is_sensitive': 'sum'
        }).reset_index()
        qtime_summary.columns = ['machine_group', 'total_ops', 'sensitive_ops']
        qtime_summary['qtime_sensitive_ratio'] = (
            qtime_summary['sensitive_ops'] / qtime_summary['total_ops']
        )
        
        result = result.merge(
            qtime_summary[['machine_group', 'qtime_sensitive_ratio']],
            on='machine_group',
            how='left'
        )
    
    # Reorder columns
    cols = ['machine_group', 'machine_count', 'total_available_operations',
            'avg_processing_time', 'total_processing_time']
    if 'qtime_sensitive_ratio' in result.columns:
        cols.append('qtime_sensitive_ratio')
    
    return result[cols]


def apply_mttf_pm_rule(
    machines_df: pd.DataFrame,
    machine_failure_df: pd.DataFrame,
    group_risk_assignment: Dict[str, str],
    risk_coefficients: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Apply MTTF-based PM rule to calculate PM intervals for each machine.
    
    This function:
    1. Calculates MTTF for each machine using Weibull parameters
    2. Assigns risk type to each machine group
    3. Calculates PM interval = coefficient × MTTF
    
    Args:
        machines_df: DataFrame with columns: machine_id, machine_group
        machine_failure_df: DataFrame with columns: machine_id, shape parameter, 
                           scale parameter, repair_time, pm_duration
        group_risk_assignment: Dict mapping machine_group -> risk_type (high/medium/low)
                              Example: {"G1": "medium", "G2": "high", ...}
        risk_coefficients: Dict mapping risk_type -> coefficient
                          Example: {"high": 0.3, "medium": 0.5, "low": 0.7}
                          If None, uses default values
    
    Returns:
        Tuple of (pm_interval_dict, machine_pm_policy, group_pm_policy):
        - pm_interval_dict: Dict mapping machine_id -> pm_interval for simulation
        - machine_pm_policy: DataFrame with machine-level PM details
        - group_pm_policy: DataFrame with group-level PM summary
    """
    if risk_coefficients is None:
        risk_coefficients = {"high": 0.3, "medium": 0.5, "low": 0.7}
    
    # Merge machines and failure data
    merged = machines_df[['machine_id', 'machine_group']].merge(
        machine_failure_df,
        on='machine_id',
        how='left'
    )
    
    # Calculate MTTF for each machine
    merged['mttf'] = merged.apply(
        lambda row: calculate_weibull_mttf(
            row['shape parameter'],
            row['scale parameter']
        ),
        axis=1
    )
    
    # Add risk type and PM coefficient
    merged['risk_type'] = merged['machine_group'].map(group_risk_assignment)
    merged['pm_coefficient'] = merged['risk_type'].map(risk_coefficients)
    
    # Calculate PM interval
    merged['pm_interval'] = merged['mttf'] * merged['pm_coefficient']
    
    # Create machine PM policy DataFrame
    machine_pm_policy = merged[[
        'machine_id',
        'machine_group',
        'shape parameter',
        'scale parameter',
        'mttf',
        'risk_type',
        'pm_coefficient',
        'pm_interval',
        'repair_time',
        'pm_duration'
    ]].copy()
    
    # Create group PM policy DataFrame
    group_pm_policy = merged.groupby('machine_group').agg({
        'machine_id': 'count',
        'shape parameter': 'mean',
        'scale parameter': 'mean',
        'mttf': 'mean',
        'repair_time': 'mean',
        'pm_duration': 'mean',
        'risk_type': 'first',
        'pm_coefficient': 'first',
        'pm_interval': 'mean'
    }).reset_index()
    
    group_pm_policy.rename(columns={'machine_id': 'machine_count'}, inplace=True)
    group_pm_policy.rename(columns={
        'shape parameter': 'avg_shape_parameter',
        'scale parameter': 'avg_scale_parameter',
        'mttf': 'avg_mttf',
        'repair_time': 'avg_repair_time',
        'pm_duration': 'avg_pm_duration',
        'pm_interval': 'avg_pm_interval'
    }, inplace=True)
    
    # Reorder columns
    group_pm_policy = group_pm_policy[[
        'machine_group',
        'machine_count',
        'avg_shape_parameter',
        'avg_scale_parameter',
        'avg_mttf',
        'avg_repair_time',
        'avg_pm_duration',
        'risk_type',
        'pm_coefficient',
        'avg_pm_interval'
    ]]
    
    # Create pm_interval_dict for simulation
    pm_interval_dict = dict(zip(
        machine_pm_policy['machine_id'],
        machine_pm_policy['pm_interval']
    ))
    
    return pm_interval_dict, machine_pm_policy, group_pm_policy


def get_pm_interval_by_rule(
    pm_rule: str,
    machines_df: pd.DataFrame,
    machine_failure_df: pd.DataFrame,
    threshold_value: Optional[float] = None,
    group_risk_assignment: Optional[Dict[str, str]] = None,
    risk_coefficients: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Get PM interval dictionary based on selected PM rule.
    
    Args:
        pm_rule: "THRESHOLD" or "MTTF"
        machines_df: Machine information DataFrame
        machine_failure_df: Machine failure data DataFrame
        threshold_value: Cumulative failure probability threshold (for THRESHOLD rule)
                        Example: 0.1, 0.3, 0.5
        group_risk_assignment: Dict for MTTF rule (group_name -> risk_type)
        risk_coefficients: Dict for MTTF rule (risk_type -> coefficient)
    
    Returns:
        Tuple of (pm_interval_dict, policy_df1, policy_df2):
        - pm_interval_dict: Machine ID -> PM interval mapping
        - policy_df1: Machine-level or group-level policy details
        - policy_df2: Summary policy details
    """
    if pm_rule.upper() == "THRESHOLD":
        if threshold_value is None:
            raise ValueError("threshold_value must be provided for THRESHOLD rule")
        
        # For THRESHOLD rule, calculate PM interval based on time at failure probability
        merged = machines_df[['machine_id', 'machine_group']].merge(
            machine_failure_df,
            on='machine_id',
            how='left'
        )
        
        merged['pm_interval'] = merged.apply(
            lambda row: weibull_time_at_failure_prob(
                row['shape parameter'],
                row['scale parameter'],
                threshold_value
            ),
            axis=1
        )
        
        pm_interval_dict = dict(zip(
            merged['machine_id'],
            merged['pm_interval']
        ))
        
        # Create summary DataFrames
        policy_df1 = merged[[
            'machine_id', 'machine_group', 'shape parameter',
            'scale parameter', 'pm_interval', 'repair_time', 'pm_duration'
        ]].copy()
        
        policy_df2 = merged.groupby('machine_group').agg({
            'machine_id': 'count',
            'pm_interval': 'mean',
            'repair_time': 'mean',
            'pm_duration': 'mean'
        }).reset_index()
        policy_df2.columns = ['machine_group', 'machine_count', 'avg_pm_interval', 
                             'avg_repair_time', 'avg_pm_duration']
        
        return pm_interval_dict, policy_df1, policy_df2
    
    elif pm_rule.upper() == "MTTF":
        if group_risk_assignment is None:
            raise ValueError("group_risk_assignment must be provided for MTTF rule")
        
        return apply_mttf_pm_rule(
            machines_df,
            machine_failure_df,
            group_risk_assignment,
            risk_coefficients
        )
    
    else:
        raise ValueError(f"Unknown PM rule: {pm_rule}. Must be 'THRESHOLD' or 'MTTF'.")
