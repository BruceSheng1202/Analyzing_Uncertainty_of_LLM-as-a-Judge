# !wget https://files.pythonhosted.org/packages/py3/R/R2CCP/R2CCP-0.0.8-py3-none-any.whl
# !pip install R2CCP-0.0.8-py3-none-any.whl --no-deps
# !pip install configargparse pytorch_lightning torchvision

import numpy as np
import random
import torch
torch.set_float32_matmul_precision('medium')

alpha = 0.10

import json
import math
import pandas as pd
from matplotlib import pyplot as plt
import os, sys


import os
os.makedirs('model_paths', exist_ok=True)

from R2CCP.main import R2CCP

import json
import pandas as pd
import math
from R2CCP.main import R2CCP
import numpy as np
import os
import random
import psutil
import time


def merge_intervals(sample_intervals):
    if not sample_intervals:
        return (1,5)
    lows = [low for low, high in sample_intervals]
    highs = [high for low, high in sample_intervals]
    return (min(lows), max(highs))

def range_modification(y_qlow, y_qup, range_low,  range_up):
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def run_experiment(X, y, seed, dimension, dataset):
    random.seed(seed)
    np.random.seed(seed)

    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)

    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    # X_cal = X_cal[:int(len(X_cal) * cal_size)]
    # y_cal = y_cal[:int(len(y_cal) * cal_size)]
    
    if os.path.exists('model_paths/model_save_destination.pth'):
        os.remove('model_paths/model_save_destination.pth')

    model = R2CCP({'model_path': 'model_paths/model_save_destination.pth', 'max_epochs': 100, 'alpha': alpha})
    model.fit(X_cal, y_cal.flatten())
    intervals = model.get_intervals(X_test)
    intervals = [merge_intervals(sample_intervals) for sample_intervals in intervals]

    df = pd.DataFrame({
        'low':    [iv[0] for iv in intervals],
        'up':     [iv[1] for iv in intervals],
        'y_test': y_test
    })

    df.to_csv(f'R2CCP_{dataset}_{dimension}_{seed}.csv', index=False)
    
    # m = 13
    # target_idx = np.linspace(3, 15, m)-3

    # adjusted_intervals = [
    # [
    #     (
    #         next((num for num in target_idx if abs(low - num) < 1/6), low),
    #         next((num for num in target_idx if abs(high - num) < 1/6), high)
    #     )
    #     for low, high in sample_intervals
    # ]
    # for sample_intervals in intervals]

    # intervals = adjusted_intervals

    in_interval = [
        (low <= y_true <= high)
        for (low, high), y_true in zip(intervals, y_test)
    ]
    coverage_rate  = np.mean(in_interval)
    average_width = np.mean([high - low for low, high in intervals])

    del model
    torch.cuda.empty_cache()
    time.sleep(1) 

    print(f"Seed: {seed}, Width: {average_width:.4f}, Coverage: {coverage_rate:.4f}")

    return average_width, coverage_rate

def calculate_statistics(X, y, num_runs=100, seed_start=1, dimension = 'consistency', dataset='summeval'):
    from tqdm import tqdm
    width = []
    coverage = []
    for i in tqdm(range(num_runs), desc="Running experiments"):
        seed = seed_start + i
        try:
            average_width, coverage_rate = run_experiment(X, y, seed, dimension, dataset)
            width.append(average_width)
            coverage.append(coverage_rate)
            print(f"Memory usage: {psutil.virtual_memory().percent}%")
        except IndexError as e:
            print(f"Skipping seed {seed} due to error: {e}")
            continue
    
    mean_width = np.mean(width)
    std_width = np.std(width)
    mean_coverage = np.mean(coverage)
    std_coverage = np.std(coverage)

    print("\nSummary of R2CCP:")
    print(f"Width: {mean_width:.4f}, {std_width:.4f}")
    print(f"Coverage: {mean_coverage:.4f}, {std_coverage:.4f}")

    return  width, coverage

folder_path = f'./model_logits/dsr1/'

dataset = 'Summeval'
data = {}
for dimension in ["consistency", "coherence", "fluency", "relevance"]:
    file_path = os.path.join(folder_path, f"Summeval_{dimension}_logits.csv")
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    run_experiment(X, y, 42, dimension, dataset)