import os
import random
import time
import numpy as np

import json
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from mapie.metrics import regression_coverage_score
from mapie.quantile_regression import MapieQuantileRegressor

def range_modification(y_qlow, y_qup, range_low,  range_up):
    y_qlow = np.clip(y_qlow, range_low, range_up)
    y_qup = np.clip(y_qup, range_low, range_up)
    return y_qlow, y_qup

def run_experiment(X, y, seed, dimension, dataset):
    random.seed(seed)
    np.random.seed(seed)

    # X = clr_transform(X)
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)

    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    gb_reg = GradientBoostingRegressor(loss="quantile", random_state=seed)
    mapie_qr = MapieQuantileRegressor(estimator=gb_reg, alpha=0.1)
    mapie_qr.fit(X_cal, y_cal, random_state=seed)

    # Symmetric prediction interval
    y_pred_sym, y_pis_sym = mapie_qr.predict(X_test, symmetry=True)
    y_pis_sym = np.clip(y_pis_sym, 1, 5)
    y_qlow = y_pis_sym[:, 0]
    y_qup = y_pis_sym[:, 1]
    coverage_sym = regression_coverage_score(y_test, y_qlow, y_qup)
    width_sym = (y_qup - y_qlow).mean()

    df = pd.DataFrame({
        'low':    y_qlow.ravel(),
        'up':     y_qup.ravel(),
        'y_test': y_test.ravel()
    })

    df.to_csv(f'CQR_sym_{dataset}_{dimension}_{seed}.csv', index=False)

    # Asymmetric prediction interval
    y_pred_asym, y_pis_asym = mapie_qr.predict(X_test, symmetry=False)
    coverage_asym = regression_coverage_score(y_test, y_pis_asym[:, 0], y_pis_asym[:, 1])
    y_pis_asym = np.clip(y_pis_asym, 1, 5)
    y_qlow = y_pis_asym[:, 0]
    y_qup = y_pis_asym[:, 1]
    coverage_asym = regression_coverage_score(y_test, y_qlow, y_qup)
    width_asym = (y_qup - y_qlow).mean()

    df = pd.DataFrame({
        'low':    y_qlow.ravel(),
        'up':     y_qup.ravel(),
        'y_test': y_test.ravel()
    })

    df.to_csv(f'CQR_asym_{dataset}_{dimension}_{seed}.csv', index=False)
    
    return coverage_sym, width_sym, coverage_asym, width_asym

def calculate_statistics(X, y, num_runs=100, seed_start=1, dimension = 'consistency', dataset='summeval'):
    from tqdm import tqdm
    width_sym_list = []
    coverage_sym_list = []
    width_asym_list = []
    coverage_asym_list = []
    for i in tqdm(range(num_runs), desc="Running experiments"):
        seed = seed_start + i
        try:
            coverage_sym, width_sym, coverage_asym, width_asym = run_experiment(X, y, seed, dimension, dataset)
            width_sym_list.append(width_sym)
            coverage_sym_list.append(coverage_sym)
            width_asym_list.append(width_asym)
            coverage_asym_list.append(coverage_asym)
        except IndexError as e:
            print(f"Skipping seed {seed} due to error: {e}")
            continue
    
    mean_width_sym = np.mean(width_sym_list)
    std_width_sym = np.std(width_sym_list)
    mean_coverage_sym = np.mean(coverage_sym_list)
    std_coverage_sym = np.std(coverage_sym_list)
    mean_width_asym = np.mean(width_asym_list)
    std_width_asym = np.std(width_asym_list)
    mean_coverage_asym = np.mean(coverage_asym_list)
    std_coverage_asym = np.std(coverage_asym_list)

    print("\nSummary of CQR sym:")
    print(f"Width: {mean_width_sym:.4f} ± {std_width_sym:.4f}")
    print(f"Coverage: {mean_coverage_sym:.4f} ± {std_coverage_sym:.4f}")
    print("\nSummary of CQR asym:")
    print(f"Width: {mean_width_asym:.4f} ± {std_width_asym:.4f}")
    print(f"Coverage: {mean_coverage_asym:.4f} ± {std_coverage_asym:.4f}")

    return  mean_width_sym, mean_coverage_sym, mean_width_asym, mean_coverage_asym

import os
import pandas as pd

folder_path = f'./model_logits/qwen/'

data = {}
for dimension in ["cosmos", "drop", "esnli", "gsm8k"]:
        file_path = os.path.join(folder_path, f"SocREval_{dimension}_logits.csv")
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        width_sym, coverage_sym, width_asym, coverage_asym = calculate_statistics(X, y, num_runs=30, seed_start=1, dimension=dimension, dataset='SocREval')

