import numpy as np
import random
import torch

alpha = 0.10
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

import json
import math
import pandas as pd
from matplotlib import pyplot as plt
import os, sys
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath("./boosted-conformal/"))
third_party_path = os.path.abspath("./boosted-conformal/third_party")
sys.path.insert(0, third_party_path)

from boostedCP.utils import cqr_preboost,local_preboost,plot_len
from boostedCP.gradient_boost import gradient_boost_len
from boostedCP.len_local_boost import len_local_boost
from boostedCP.len_cqr_boost import len_cqr_boost

import json
import pandas as pd
import numpy as np
import random
import time

def range_modification(y_qlow, y_qup, range_low,  range_up):
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def run_experiment(X, y, seed, type = 'cqr', dataset = 'summeval', dimension = 'consistency'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32).ravel()

    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    if type == 'lcp':
        results = len_local_boost(X_cal, y_cal, X_cal, y_cal,X_test, y_test, 
                                alpha, seed, n_rounds_cv=500, learning_rate=0.02, store=True, verbose=False)
    if type == 'cqr':
        results = len_cqr_boost(X_cal, y_cal, X_cal, y_cal,X_test, y_test, 
                                alpha, seed, n_rounds_cv=500, learning_rate=0.02, store=True, verbose=False)
    
    y_qlow, y_qup = range_modification(results["boosted_lower"].flatten(), results["boosted_upper"].flatten(), 1, 5)
    intervals = [
    [(low, high)] for low, high in zip(y_qlow, y_qup)
    ]

    df = pd.DataFrame({
        'low':    [iv[0][0] for iv in intervals],
        'up':     [iv[0][1] for iv in intervals],
        'y_test': y_test
    })

    df.to_csv(f'Boosted_{type}_{dataset}_{dimension}_{seed}.csv', index=False)

    adjusted_intervals = [
    [
        (
            next((num for num in [1, 2, 3, 4, 5] if abs(low - num) < 0.1), low),
            next((num for num in [1, 2, 3, 4, 5] if abs(high - num) < 0.1), high)
        )
        for low, high in sample_intervals
    ]
    for sample_intervals in intervals]

    intervals = adjusted_intervals

    in_interval = [
        any(low <= true_value <= high for low, high in sample_intervals)
        for sample_intervals, true_value in zip(intervals, y_test)
    ]

    coverage_rate = np.mean(in_interval)
    average_width = np.mean([high - low for sample_intervals in intervals for low, high in sample_intervals])  

    return average_width, coverage_rate

def calculate_statistics(X, y, num_runs=30, seed_start=1, dataset='summeval', dimension='consistency'):
    from tqdm import tqdm
    width_lcp = []
    coverage_lcp = []
    width_cqr = []
    coverage_cqr = []
    for i in tqdm(range(num_runs), desc="Running experiments"):
        seed = seed_start + i 
        average_width, coverage_rate = run_experiment(X, y, seed, 'lcp', dataset, dimension)
        width_lcp.append(average_width)
        coverage_lcp.append(coverage_rate)

        average_width, coverage_rate = run_experiment(X, y, seed, 'cqr', dataset, dimension)
        width_cqr.append(average_width)
        coverage_cqr.append(coverage_rate)

    mean_width_lcp = np.mean(width_lcp)
    std_width_lcp = np.std(width_lcp)
    mean_coverage_lcp = np.mean(coverage_lcp)
    std_coverage_lcp = np.std(coverage_lcp)
    mean_width_cqr = np.mean(width_cqr)
    std_width_cqr = np.std(width_cqr)
    mean_coverage_cqr = np.mean(coverage_cqr)
    std_coverage_cqr = np.std(coverage_cqr)
    
    print("\nSummary of Boosted LCP:")
    print(f"Width: {mean_width_lcp:.4f}, {std_width_lcp:.4f}")
    print(f"Coverage: {mean_coverage_lcp:.4f}, {std_coverage_lcp:.4f}")

    print("\nSummary of Boosted CQR:")
    print(f"Width: {mean_width_cqr:.4f}, {std_width_cqr:.4f}")
    print(f"Coverage: {mean_coverage_cqr:.4f}, {std_coverage_cqr:.4f}")
    
    return width_lcp, coverage_lcp, width_cqr, coverage_cqr

import os
import pandas as pd

folder_path = f'./model_logits/qwen/'
for dimension in ["cosmos", "drop", "esnli", "gsm8k"]:
        file_path = os.path.join(folder_path, f"SocREval_{dimension}_logits.csv")
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        width_lcp, coverage_lcp, width_cqr, coverage_cqr = calculate_statistics(X, y, num_runs=30, seed_start=1, dimension=dimension, dataset='SocREval')

