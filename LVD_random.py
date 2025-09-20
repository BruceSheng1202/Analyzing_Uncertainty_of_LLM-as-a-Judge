# !git clone https://github.com/zlin7/LVD.git
# %cd LVD
# !pip install ipdb


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

import json
import pandas as pd
import math
from data.preprocess_small_datasets import pretrain_general
from models.regmodel import KernelMLKR
from models.conformal import LocalConditional, PIConstructor
import numpy as np
import os
import random
import psutil
import time


def range_modification(y_qlow, y_qup, range_low,  range_up):
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def run_experiment(X, y, seed, dataset, dimension):
    random.seed(seed)
    np.random.seed(seed)

    # X = clr_transform(X)
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)

    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    # X_cal = X_cal[:int(len(X_cal) * cal_size)]
    # y_cal = y_cal[:int(len(y_cal) * cal_size)]
    
    y_cal = y_cal.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    DNN_model, readout_layer = pretrain_general(X_cal, y_cal, seed=0, quiet=True, model_setting=0)
    embed_cal = DNN_model.embed(X_cal)
    embed_test = DNN_model.embed(X_test)

    kernel_model = KernelMLKR(d=10, seed=0, n_iters=500, norm=True, lr=1e-3)
    kernel_model.fit(embed_cal, y_cal.flatten()) 

    lvd = LocalConditional(K_obj=kernel_model)
    lvd.fit(embed_cal, y_cal.flatten(), m=readout_layer)

    results = lvd.eval(embed_test, y_test.flatten(), lvd.PI, alpha=alpha, quiet=False)

    y_qlow = results['lo'].to_numpy()
    y_qup = results['hi'].to_numpy()
    y_qlow, y_qup = range_modification(y_qlow, y_qup, 1, 5)

    intervals = [[(low, high)] for low, high in zip(y_qlow, y_qup)]
    print(intervals)
    
    df = pd.DataFrame({
        'low':    [iv[0][0] for iv in intervals],
        'up':     [iv[0][1] for iv in intervals],
        'y_test': y_test.flatten()
    })
    
    df.to_csv(f'LVD_{dataset}_{dimension}_{seed}.csv', index=False)

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

    print(f"Seed: {seed}, Width: {average_width:.4f}, Coverage: {coverage_rate:.4f}")

    return average_width, coverage_rate

def calculate_statistics(X, y, num_runs=100, seed_start=1, dataset = 'Summeval', dimension = 'consistency'):
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

    print("\nSummary of LVD:")
    print(f"Width: {mean_width:.4f}, {std_width:.4f}")
    print(f"Coverage: {mean_coverage:.4f}, {std_coverage:.4f}")

    return  width, coverage

dimension = 'consistency'
dataset = 'Dialsumm'

# folder_path = f'../data_results/prompt_logits/data_logits/{dataset}/'
# file_path = os.path.join(folder_path, f"{dataset}_{dimension}.csv")
folder_path = f'../model_logits/qwen/'
file_path = os.path.join(folder_path, f"{dataset}_{dimension}_logits.csv")
df = pd.read_csv(file_path)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

width, coverage = calculate_statistics(X, y, 30, 1, dataset, dimension)