# !git clone https://github.com/msesia/chr.git
# %cd chr
# !pip install rpy2
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# There are two edits in the chr/black_boxes.py, otherwise the code will not run.
#Edit the black_boxes.py: 'import torch.tensor as tensor' → 'from torch import tensor'
#Edit the black_boxes.py: 'from skgarden import RandomForestQuantileRegressor' → 'from sklearn.linear_model import QuantileRegressor'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import pdb
import torch

alpha = 0.10
print("Is CUDA available? {}".format(torch.cuda.is_available()))

from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')

import json
import pandas as pd
import math
import numpy as np
import os
import random

from chr.black_boxes import QNet
from chr.histogram import Histogram
from chr.grey_boxes import HistogramAccumulator



def range_modification(y_qlow, y_qup, range_low,  range_up):
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def run_experiment(X, y, seed, dataset='Summeval', dimension='consistency'):
    random.seed(seed)
    np.random.seed(seed)

    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)

    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    grid_quantiles = np.arange(0.01, 1.0, 0.01)

    bbox = QNet(
        grid_quantiles,
        X_cal.shape[1],
        no_crossing=True,
        batch_size=32,
        dropout=0.1,
        num_epochs=1000,
        learning_rate=0.0005,
        num_hidden=256,
        calibrate=False
    )

    bbox.fit(X_cal, y_cal)

    grid_histogram = np.arange(0, 6, 0.1)

    hist = Histogram(grid_quantiles, grid_histogram)

    Q_test = bbox.predict(X_test)

    histogram_test = hist.compute_histogram(Q_test, 0, 6, 0.001)

    accumulator = HistogramAccumulator(histogram_test, grid_histogram, alpha=alpha, delta_alpha=0.01)

    epsilon = np.random.uniform(low=0.0, high=1.0, size=X_test.shape[0])
    S, bands = accumulator.predict_intervals(alpha, epsilon=epsilon)
    S_int = [np.arange(S[i][0],S[i][1]+1) for i in range(len(S))]
    intervals_crch = np.array([[grid_histogram[S_int[i]-1][0],grid_histogram[S_int[i]][-1]] for i in range(len(S_int))])

    y_qlow = np.min(intervals_crch, axis=1)
    y_qup = np.max(intervals_crch, axis=1)

    y_qlow, y_qup = range_modification(y_qlow, y_qup, 1, 5)
    intervals = [
    [(low, high)] for low, high in zip(y_qlow, y_qup)
]

    df = pd.DataFrame({
        'low':    [iv[0][0] for iv in intervals],
        'up':     [iv[0][1] for iv in intervals],
        'y_test': y_test
    })

    df.to_csv(f'CHR_{dataset}_{dimension}_{seed}.csv', index=False)

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

def calculate_statistics(X, y, num_runs=100, seed_start=1, dataset='Summeval', dimension='consistency'):
    from tqdm import tqdm
    width = []
    coverage = []
    for i in tqdm(range(num_runs), desc="Running experiments"):
        seed = seed_start + i
        average_width, coverage_rate = run_experiment(X, y, seed, dataset, dimension)
        width.append(average_width)
        coverage.append(coverage_rate)

    mean_width = np.mean(width)
    std_width = np.std(width)
    mean_coverage = np.mean(coverage)
    std_coverage = np.std(coverage)

    print("\nSummary of CHR:")
    print(f"Width: {mean_width:.4f}, {std_width:.4f}")
    print(f"Coverage: {mean_coverage:.4f}, {std_coverage:.4f}")

    return  width, coverage

dimension = 'relevance'
dataset = 'Summeval'

# folder_path = f'../data_results/prompt_logits/data_logits/{dataset}/'
# file_path = os.path.join(folder_path, f"{dataset}_{dimension}.csv")
folder_path = f'../model_logits/qwen/'
file_path = os.path.join(folder_path, f"{dataset}_{dimension}_logits.csv")
df = pd.read_csv(file_path)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

width, coverage = calculate_statistics(X, y, 1, 1, dataset, dimension)