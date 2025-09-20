import pandas as pd
import numpy as np

def range_modification(y_qlow, y_qup, range_low=1, range_up=5):
    """
    range clipping
    """
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def boundary_adjustment(value, label_set, threshold=0.1):
    """
    adjust to nearest valid label
    """
    threshold_max = (label_set[-1] - label_set[0]) / (len(label_set) - 1) / 2
    threshold = min(threshold_max, threshold)
    adjusted_value = next((num for num in label_set if abs(num - value) < threshold), value)
    return adjusted_value

def calculate_coverage_and_width(low, up, y_test):
    """
    calculate coverage and width
    """
    low = np.asarray(low)
    up = np.asarray(up)
    y_test = np.asarray(y_test)
    
    width = up - low
    coverage = np.mean((low <= y_test.flatten()) & (y_test.flatten() <= up))
    return width.mean(), coverage.mean()
    
def calculate_midpoints(df):
    """
    calculate midpoints
    """
    df['midpoint'] = (df['lb'] + df['ub']) / 2
    df['midpoint_adj'] = (df['lb_adjusted'] + df['ub_adjusted']) / 2
    return df
