import numpy as np
import random
import torch

alpha = 0.10

import json
import math
import pandas as pd
from matplotlib import pyplot as plt
import os, sys

from abc import ABC, abstractmethod
import numpy as np

# Base class for Ordinal Regression Ordinal Risk Predictor;
# a typical calling order is: find_lambda -> calc_loss -> get_prediction_set_bounds;
class OrdinalRegCRPredictor:

    # calculate the bound [l, u] for the optimal prediction set
    # such that sum of sy between the bound is greater than the lambda value.
    @abstractmethod
    def get_prediction_set_bounds(self, fyx, lambda_val):
        pass

    # calculate the incurred loss for a specific record
    @abstractmethod
    def calc_loss(self, fyx, y, lambda_val):
        pass

    # find the optimal value of lambda for a dataset
    # such that the risk on this dataset is controlled by alpha
    @abstractmethod
    def find_lambda(self, fyxs, ys, alpha):
        pass

    # run prediction for a batch of records
    @abstractmethod
    def run_predictions(self, fyxs, ys, lambda_val):
        pass

class WeightedCRPredictor(OrdinalRegCRPredictor):

    # initialize the weights
    # normalize the weights so that the maximal value is 1
    def __init__(self, hy):
        max_hy = np.max(hy)
        self.hy = hy / max_hy
        self.num_classes = hy.size

    # get the prediction set for given fyx and lambda_val
    # this implementation is greedy
    # it starts at the index with the largest sy=hy*fyx value
    # then gradually extends the boundary until the risk meets the required limit
    # after that, it tries to shrinks the boundary to squeeze the ones with zero risks
    # the last step is needed to avoid producing a too large prediction set
    def get_prediction_set_bounds(self, fyx, lambda_val):
        sy = self.hy * fyx

        #b_val = sum(sy) ## conditional
        b_val = 1 ## marginal
        threshold = b_val - lambda_val

        index_max = np.argmax(sy)
        s = sy[index_max]
        l, u = index_max, index_max
        while s < threshold:
            if l - 1 >= 0 and u + 1 <= self.num_classes -1:
                if sy[l - 1] >= sy[u + 1]:
                    l = l - 1
                    s = s + sy[l]
                else:
                    u = u + 1
                    s = s + sy[u]
            elif l - 1 >= 0:
                l = l - 1
                s = s + sy[l]
            elif u + 1 <= self.num_classes - 1:
                u = u + 1
                s = s + sy[u]
            else:
                break

        while sy[l] == 0 and l < u:
            l = l + 1
        while sy[u] == 0 and l < u:
            u = u - 1

        return l, u

    # calculate the incurred loss for a specific record
    # hy:         weights of different labels;
    # fyx:        model scores of different labels;
    # y:          true label;
    # lambda_val: a proposed risk bound;
    def calc_loss(self, fyx, y, lambda_val):
        lower_bound, upper_bound = self.get_prediction_set_bounds(fyx, lambda_val)
        if (y >= lower_bound) and (y <= upper_bound):
            return 0.0
        else:
            return self.hy[int(y)]

    # find the optimal value of lambda such that the risk is controlled by alpha
    # fyxs:  the matrix of model scores, where each row is for one record, each column is for one class;
    # ys:    the array of true labels;
    # alpha: risk bound, value between 0 and 1;
    def find_lambda(self, fyxs, ys, alpha):
        (num_records, num_classes) = fyxs.shape
        b_val = 1 #max(hy)
        threshold = (num_records + 1.0) / num_records * alpha - b_val / num_records

        cur_lambda = 0.5
        delta = 0.5
        delta_threshold = 0.0005
        while delta > delta_threshold:
            total_r = 0.0
            for i in range(num_records):
                total_r = total_r + self.calc_loss(fyxs[i, :], ys[i], cur_lambda)
            avg_r = total_r / num_records
            if avg_r > threshold:
                cur_lambda = cur_lambda - delta / 2
            elif avg_r < threshold:
                cur_lambda = cur_lambda + delta / 2
            else:
                break
            delta = delta / 2
        return cur_lambda

    def run_predictions(self, fyxs, ys, lambda_val):
        num_records = fyxs.shape[0]
        prediction_sets = []
        losses = []
        for i in range(num_records):
            lower_bound, upper_bound = self.get_prediction_set_bounds(fyxs[i, :], lambda_val)
            prediction_sets.append((lower_bound, upper_bound))
            label = int(ys[i])
            if (label >= lower_bound) and (label <= upper_bound):
                losses.append(0.0)
            else:
                losses.append(self.hy[label])
        return prediction_sets, losses

class DivergenceCRPredictor(OrdinalRegCRPredictor):

    # Given fyx, calculate the cumulative head scores
    # head_j = sum_{i=0}^{j}(fyx_i) / (K-1)
    def get_head_scores(self, fyx):
        num_classes = fyx.size
        head_scores = np.zeros(num_classes)
        prev = 0
        for i in range(num_classes):
            head_scores[i] = prev + fyx[i]
            prev = head_scores[i]
        return [v / (num_classes - 1) for v in head_scores]


    # Given fyx, calculate the cumulative tail scores
    # tail_j = sum_{i=j}^{K-1}(fyx_i) / (K-1)
    def get_tail_scores(self, fyx):
        num_classes = fyx.size
        tail_scores = np.zeros(num_classes)
        prev = 0
        for i in range(num_classes - 1, -1, -1):
            tail_scores[i] = prev + fyx[i]
            prev = tail_scores[i]
        return [v / (num_classes - 1) for v in tail_scores]

    # Given fyx, as well as a threshold,
    # find out the optimal bound [y_l, y_u] of the prediction set,
    # such that the total risk is less than the threshold.
    # this implementation is greedy.
    def get_prediction_set_bounds(self, fyx, lambda_val):
        num_classes = fyx.size
        head_scores = self.get_head_scores(fyx)
        tail_scores = self.get_tail_scores(fyx)

        l, u = np.argmax(fyx), np.argmax(fyx)
        sums = np.zeros(num_classes)
        steps = [None] * num_classes
        steps[0] = (l, u)
        s = 0
        for i in range(num_classes - 1):
            if l > 0 and u < num_classes - 1:
                if head_scores[l - 1] >= tail_scores[u + 1]:
                    s = s + head_scores[l - 1]
                    l = l - 1
                else:
                    s = s + tail_scores[u + 1]
                    u = u + 1
            elif l == 0:
                s = s + tail_scores[u + 1]
                u = u + 1
            else:
                s = s + head_scores[l - 1]
                l = l - 1
            sums[i + 1] = s
            steps[i + 1] = (l, u)

        for i in range(num_classes):
            if sums[num_classes - 1] - sums[i] <= lambda_val:
                l, u = steps[i][0], steps[i][1]
                break

        return l, u

    # calculate the incurred loss for a specific record
    # fyx:        model scores of different labels;
    # y:          true label;
    # lambda_val: a proposed risk bound;
    def calc_loss(self, fyx, y, lambda_val):
        num_classes = fyx.size
        lower_bound, upper_bound = self.get_prediction_set_bounds(fyx, lambda_val)
        if y < lower_bound:
            return (lower_bound - y) / (num_classes - 1)
        elif y > upper_bound:
            return (y - upper_bound) / (num_classes - 1)
        else:
            return 0.0

    # find the optimal value of lambda such that the risk is controlled by alpha
    # fyxs:  the matrix of model scores, where each row is for one record, each column is for one class;
    # ys:    the array of true labels;
    # alpha: risk bound, value between 0 and 1;
    def find_lambda(self, fyxs, ys, alpha):
        (num_records, num_classes) = fyxs.shape
        b_val = 1
        threshold = (num_records + 1.0) / num_records * alpha - b_val / num_records

        cur_lambda = 0.5
        delta = 0.5
        delta_threshold = 0.0001
        while delta > delta_threshold:
            total_r = 0.0
            for i in range(num_records):
                total_r = total_r + self.calc_loss(fyxs[i, :], ys[i], cur_lambda)
            avg_r = total_r / num_records
            if avg_r > threshold:
                cur_lambda = cur_lambda - delta / 2
            elif avg_r < threshold:
                cur_lambda = cur_lambda + delta / 2
            else:
                break
            delta = delta / 2
        return cur_lambda

    def run_predictions(self, fyxs, ys, lambda_val):
        num_records = fyxs.shape[0]
        num_classes = fyxs.shape[1]
        prediction_sets = []
        losses = []
        for i in range(num_records):
            lower_bound, upper_bound = self.get_prediction_set_bounds(fyxs[i, :], lambda_val)
            prediction_sets.append((lower_bound, upper_bound))
            label = int(ys[i])
            if label < lower_bound:
                losses.append((lower_bound - label) / (num_classes - 1))
            elif label > upper_bound:
                losses.append((label - upper_bound) / (num_classes - 1))
            else:
                losses.append(0.0)
        return prediction_sets, losses
    
def y_data_load(filename, dimension):
  with open(filename, 'r') as file:
    content = file.read()
  lines = content.splitlines()
  data = []
  for line in lines:
    if line.strip():
        try:
            json_obj = json.loads(line)
            data.append({'custom_id': json_obj['custom_id'], dimension: json_obj[dimension]})
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")
  y = pd.DataFrame(data)
  y.set_index('custom_id', inplace=True)
  y = y[dimension]
  return y

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def run_experiment(X, y, seed, dataset, dimension):
    random.seed(seed)
    np.random.seed(seed)

    X.columns = list(range(len(X.columns)))
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)-1

    # labels = np.array([1, 4/3, 5/3, 2, 7/3, 8/3, 3, 10/3, 11/3, 4, 13/3, 14/3, 5])
    # y = y*3
    # from scipy.interpolate import interp1d
    # n = X.shape[0]
    # m = 13

    # new_x = np.zeros((n, m))
    # orig_idx = np.linspace(3, 15, 5)-3
    # target_idx = np.linspace(3, 15, m)-3

    # for i in range(n):
    #     f = interp1d(orig_idx, X[i, :], kind='linear')
    #     new_x[i, :] = f(target_idx)
    
    # X = softmax(new_x)   

    # from sklearn.model_selection import train_test_split

    # fyxs_cal, fyxs_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    # y_cal = y_cal.ravel()
    # y_test = y_test.ravel()

    # hy = np.ones(13)

    # predictor = WeightedCRPredictor(hy)
    # optimal_lambda = predictor.find_lambda(fyxs_cal, y_cal, alpha)
    # prediction_sets, losses = predictor.run_predictions(fyxs_test, y_test, optimal_lambda)
    # # predictor = DivergenceCRPredictor()
    # # optimal_lambda = predictor.find_lambda(fyxs_cal, y_cal, alpha)
    # # prediction_sets, losses = predictor.run_predictions(fyxs_test, y_test, optimal_lambda)

    # y_qlow = np.array([interval[0] for interval in prediction_sets])/3+1
    # y_qup = np.array([interval[1] for interval in prediction_sets])/3+1

    # y_test_real = y_test/3+1

    # labels = np.array([1, 2, 3, 4, 5])
    from sklearn.model_selection import train_test_split
    X = softmax(X)
    fyxs_cal, fyxs_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    y_cal = y_cal.ravel()
    y_test = y_test.ravel()

    hy = np.ones(5)

    predictor = WeightedCRPredictor(hy)
    optimal_lambda = predictor.find_lambda(fyxs_cal, y_cal, alpha)
    prediction_sets, losses = predictor.run_predictions(fyxs_test, y_test, optimal_lambda)

    # predictor = DivergenceCRPredictor()
    # optimal_lambda = predictor.find_lambda(fyxs_cal, y_cal, alpha)
    # prediction_sets, losses = predictor.run_predictions(fyxs_test, y_test, optimal_lambda)

    y_qlow = np.array([interval[0] for interval in prediction_sets])+1
    y_qup = np.array([interval[1] for interval in prediction_sets])+1

    y_test_real = y_test+1

    df = pd.DataFrame({
        'low':    y_qlow.ravel(),
        'up':     y_qup.ravel(),
        'y_test': y_test_real.ravel(),
    })

    df.to_csv(f'OrdinalRC_{dataset}_{dimension}_{seed}.csv', index=False)

    intervals = [[(low, high)] for low, high in zip(y_qlow, y_qup)]

    in_interval = [
        any(low <= true_value <= high for low, high in sample_intervals)
        for sample_intervals, true_value in zip(intervals, y_test_real)
    ]

    coverage_rate = np.mean(in_interval)
    average_width = np.mean([high - low for sample_intervals in intervals for low, high in sample_intervals])  

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

    print("\nSummary of Ordinal Risk Control:")
    print(f"Width: {mean_width:.4f}, {std_width:.4f}")
    print(f"Coverage: {mean_coverage:.4f}, {std_coverage:.4f}")

    return  width, coverage


import os
import pandas as pd

# folder_path = './data_results/prompt_logits/data_logits/Socreval'
folder_path = f'./model_logits/qwen/'

data = {}
for dimension in ["cosmos", "drop", "esnli", "gsm8k"]:
        file_path = os.path.join(folder_path, f"SocREval_{dimension}_logits.csv")
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        width, coverage = calculate_statistics(X, y, num_runs=30, seed_start=1, dimension=dimension, dataset='SocREval')


