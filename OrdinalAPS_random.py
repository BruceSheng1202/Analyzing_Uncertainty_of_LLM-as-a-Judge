import numpy as np
import random
import torch

alpha = 0.10

import json
import math
import pandas as pd
from matplotlib import pyplot as plt
import os, sys

def get_qhat_ordinal_aps(prediction_function, cal_scores, cal_labels, alpha):
    n = cal_scores.shape[0]
    grid_size = 10000
    for q in np.linspace(1e-3, 1 - 1e-3, grid_size)[::-1]:
        coverage, _, _ = evaluate_sets(prediction_function, np.copy(cal_scores), np.copy(cal_labels), q, alpha)
        if coverage <= (np.ceil((n + 1)*(1 - alpha))/n):
            # return q + 1/(grid_size - 1)
            return np.minimum(q + 1/(grid_size - 1), 1.0 - 1e-6)  # Clip q to be less than 1.0
    return q

def evaluate_sets(prediction_function, val_scores, val_labels, qhat, alpha, print_bool=False):
    sets = prediction_function(val_scores, qhat)
    # Check
    sizes = sets.sum(axis=1)
    sizes_distribution = np.array([(sizes == i).mean() for i in range(5)])
    # Evaluate coverage
    covered = sets[np.arange(val_labels.shape[0]), val_labels]
    coverage = covered.mean()
    label_stratified_coverage = [
        covered[val_labels == j].mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    label_distribution = [
        (val_labels == j).mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    if(print_bool):
        print(r'$\alpha$' + f":{alpha}  |  coverage: {coverage}  |  average size: {sizes.mean()}  |  qhat: {qhat}  |  set size distribution: {sizes_distribution} ")
        print(f"label stratified coverage: {label_stratified_coverage}  \nlabel distribution: {label_distribution}")
    return coverage, label_stratified_coverage, sizes_distribution

def ordinal_aps_prediction(val_scores, qhat):
    import numpy as np

    n_samples, n_classes = val_scores.shape
    P = val_scores == val_scores.max(axis=1)[:, None]

    idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat

    max_iter = n_classes  
    iter_count = 0

    while idx_construction_incomplete.sum() > 0:
        iter_count += 1
        if iter_count > max_iter:
            P[idx_construction_incomplete] = True
            break

        P_inc = P[idx_construction_incomplete]
        scores_inc = val_scores[idx_construction_incomplete]

        set_cumsum = P_inc.cumsum(axis=1)
        lower_edge_idx = (P_inc > 0).argmax(axis=1)
        upper_edge_idx = set_cumsum.argmax(axis=1)

        left_valid = (lower_edge_idx - 1) >= 0
        right_valid = (upper_edge_idx + 1) < scores_inc.shape[1]

        lower_edge_wins = np.zeros(scores_inc.shape[0], dtype=bool)

        lower_edge_wins[~right_valid & left_valid] = True

        both_valid = left_valid & right_valid
        lower_scores = scores_inc[np.arange(scores_inc.shape[0])[both_valid], lower_edge_idx[both_valid] - 1]
        upper_scores = scores_inc[np.arange(scores_inc.shape[0])[both_valid], upper_edge_idx[both_valid] + 1]
        lower_edge_wins[both_valid] = lower_scores > upper_scores

        valid_left = lower_edge_wins & ((lower_edge_idx - 1) >= 0)
        P_inc[valid_left, lower_edge_idx[valid_left] - 1] = True

        valid_right = (~lower_edge_wins) & ((upper_edge_idx + 1) < scores_inc.shape[1])
        P_inc[valid_right, upper_edge_idx[valid_right] + 1] = True

        P[idx_construction_incomplete] = P_inc

        idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat

    return P

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def run_experiment(X, y, seed, dataset='summeval', dimension='consistency'):
    random.seed(seed)
    np.random.seed(seed)

    X.columns = list(range(len(X.columns)))
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)*3-3

    y = y.astype(int)
    x_arr = X
    from scipy.interpolate import interp1d
    n = x_arr.shape[0]
    m = 13

    new_x = np.zeros((n, m))
    orig_idx = np.linspace(3, 15, 5)-3
    target_idx = np.linspace(3, 15, m)-3

    for i in range(n):
        f = interp1d(orig_idx, x_arr[i, :], kind='linear')
        new_x[i, :] = f(target_idx)
    
    new_x = softmax(new_x)
    # new_x = softmax(x_arr)

    from sklearn.model_selection import train_test_split

    fyxs_cal, fyxs_test, y_cal, y_test = train_test_split(new_x, y, test_size=0.5, random_state=seed)
    y_cal = y_cal.ravel()
    y_test = y_test.ravel()

    cal_scores = fyxs_cal
    cal_labels = y_cal
    test_scores = fyxs_test
    test_labels = y_test

    qhat = get_qhat_ordinal_aps(ordinal_aps_prediction, np.copy(cal_scores), np.copy(cal_labels), alpha)
    test_pred_sets = ordinal_aps_prediction(np.copy(test_scores), qhat)
    prediction_intervals = []
    for pred_set in test_pred_sets:
        indices = np.where(pred_set)[0]
        if len(indices) > 0:
            interval = (indices.min(), indices.max())
        else:
            interval = None 
        prediction_intervals.append(interval)

    y_qlow, y_qup = zip(*prediction_intervals)
    y_qlow = (np.array(y_qlow)+3)/3
    y_qup = (np.array(y_qup)+3)/3

    y_test_real = test_labels/3+1
    # y_qlow = np.array(y_qlow)+1
    # y_qup = np.array(y_qup)+1

    # y_test_real = test_labels+1

    df = pd.DataFrame({
        'low':    y_qlow.ravel(),
        'up':     y_qup.ravel(),
        'y_test': y_test_real.ravel(),
    })

    df.to_csv(f'OrdinalAPS_{dataset}_{dimension}_{seed}.csv', index=False)

    in_interval = (y_test_real >= y_qlow) & (y_test_real <= y_qup)

    average_width = np.mean(y_qup-y_qlow)
    coverage_rate = np.mean(in_interval)

    print(f"Seed: {seed}, Width: {average_width:.4f}, Coverage: {coverage_rate:.4f}")

    return average_width, coverage_rate

def calculate_statistics(X, y, num_runs=100, seed_start=1, dataset='summeval', dimension='consistency'):
    from tqdm import tqdm
    width = []
    coverage = []
    for i in tqdm(range(num_runs), desc="Running experiments"):
        seed = seed_start + i
        try:
            average_width, coverage_rate = run_experiment(X, y, seed, dataset, dimension)
            width.append(average_width)
            coverage.append(coverage_rate)
        except IndexError as e:
            print(f"Skipping seed {seed} due to error: {e}")
            continue
    
    mean_width = np.mean(width)
    std_width = np.std(width)
    mean_coverage = np.mean(coverage)
    std_coverage = np.std(coverage)

    print("\nSummary of Ordinal APS:")
    print(f"Width: {mean_width:.4f}, {std_width:.4f}")
    print(f"Coverage: {mean_coverage:.4f}, {std_coverage:.4f}")

    return  width, coverage

import os
import pandas as pd

folder_path = f'./model_logits/qwen/'
dataset = 'Dialsumm'
data = {}
for dimension in ["consistency", "coherence", "fluency", "relevance"]:
        file_path = os.path.join(folder_path, f"{dataset}_{dimension}_logits.csv")
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        width, coverage = calculate_statistics(X, y, num_runs=30, seed_start=1, dimension=dimension, dataset=dataset)

