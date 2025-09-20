import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_correlations(df, human_ratings_col, scores_cols):
    """
    calculate Pearson, Spearman, and Kendall correlation coefficients.
    """
    results = {}
    for col in scores_cols:
        pearson_corr, _ = pearsonr(df[human_ratings_col], df[col])
        spearman_corr, _ = spearmanr(df[human_ratings_col], df[col])
        kendall_corr, _ = kendalltau(df[human_ratings_col], df[col])
        results[col] = {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr
        }
    return results

def calculate_errors(df, human_ratings_col, scores_cols):
    """
    calculate MSE, MAE, and RMSE.
    """
    metrics = {}
    for col in scores_cols:
        mse = mean_squared_error(df[human_ratings_col], df[col])
        mae = mean_absolute_error(df[human_ratings_col], df[col])
        rmse = mse ** 0.5
        metrics[col] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
    return metrics

if __name__ == '__main__':

    from data_processing import load_data, extract_features, calculate_weighted_score
    file_path = '../evaluations/CoT_results.json'
    df = load_data(file_path)
    df = extract_features(df)
    df = calculate_weighted_score(df)
    
    scores_to_evaluate = ['raw_score', 'weighted_score']
    
    correlation_results = calculate_correlations(df, 'human_ratings', scores_to_evaluate)
    print("Correlation Results:")
    print(correlation_results)
    
    error_metrics = calculate_errors(df, 'human_ratings', scores_to_evaluate)
    print("\nError Metrics:")
    print(error_metrics)