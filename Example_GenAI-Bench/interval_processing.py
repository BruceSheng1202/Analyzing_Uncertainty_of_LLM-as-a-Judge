import pandas as pd
import numpy as np

def range_modification(y_qlow, y_qup, range_low=1, range_up=5):
    """
    将区间的上下界裁剪到指定范围内。
    """
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def boundary_adjustment(value, label_set, threshold=0.1):
    """
    将预测值调整到最近的合法评分。
    """
    threshold_max = (label_set[-1] - label_set[0]) / (len(label_set) - 1) / 2
    threshold = min(threshold_max, threshold)
    adjusted_value = next((num for num in label_set if abs(num - value) < threshold), value)
    return adjusted_value

def calculate_coverage_and_width(low, up, y_test):
    """
    计算预测区间的平均宽度和覆盖率。
    """
    low = np.asarray(low)
    up = np.asarray(up)
    y_test = np.asarray(y_test)
    
    width = up - low
    coverage = np.mean((low <= y_test.flatten()) & (y_test.flatten() <= up))
    return width.mean(), coverage.mean()
    
def calculate_midpoints(df):
    """
    计算预测区间的中间点。
    """
    df['midpoint'] = (df['lb'] + df['ub']) / 2
    df['midpoint_adj'] = (df['lb_adjusted'] + df['ub_adjusted']) / 2
    return df

if __name__ == '__main__':
    # 示例用法 (需要先运行conformal_prediction.py得到intervals)
    from data_processing import load_data, extract_features
    from conformal_prediction import train_and_get_intervals
    
    file_path = '../evaluations/CoT_results.json'
    df = load_data(file_path)
    df = extract_features(df)
    
    X = df[['1', '2', '3', '4', '5']].to_numpy().astype(np.float32)
    y = df[['human_ratings']].to_numpy().astype(np.float32)
    
    X1, X2 = X[:4800], X[4800:]
    y1, y2 = y[:4800], y[4800:]
    
    intervals1, intervals2 = train_and_get_intervals(X1, y1, X2, y2)
    full_intervals = np.array(intervals1 + intervals2)
    
    # 区间处理
    intervals_modified = np.array([range_modification(low, up) for low, up in full_intervals])
    df['lb'] = intervals_modified[:, 0]
    df['ub'] = intervals_modified[:, 1]
    
    list_scores = [1, 1.33, 1.67, 2, 2.33, 2.67, 3, 3.33, 3.67, 4, 4.33, 4.67, 5]
    df['lb_adjusted'] = df['lb'].apply(lambda x: boundary_adjustment(x, list_scores, threshold=0.167))
    df['ub_adjusted'] = df['ub'].apply(lambda x: boundary_adjustment(x, list_scores, threshold=0.167))
    
    # 计算并打印覆盖率和宽度
    width, coverage = calculate_coverage_and_width(df['lb'], df['ub'], df['human_ratings'])
    print(f'Average Width: {width}, Coverage: {coverage}')
    
    # 计算中间点
    df = calculate_midpoints(df)
    print("\nSample DataFrame with midpoints:")
    print(df.head())