import glob
import os
import pandas as pd
from scipy.stats import binomtest

results = []

for tmp in ['tmp0', 'tmp1']:
    for model in ['qwen', 'dsr1', '4omini']:
            for dimension in ['consistency', 'coherence', 'fluency', 'relevance']:
                coverages = []
                widths = []
                for seed in range(31):
                    csv_path = f'./{model}_{tmp}/R2CCP_Summeval_{dimension}_{seed}.csv'
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        df = df.rename(columns={'low': 'y_qlow', 'up': 'y_qup'})
                        df['y_test'] = round(df['y_test'], 2)
                        df['y_qlow'] = round(df['y_qlow'], 2)
                        df['y_qup'] = round(df['y_qup'], 2)
                        coverage = ((df['y_test'] >= df['y_qlow']) & (df['y_test'] <= df['y_qup'])).mean()
                        width = (df['y_qup'] - df['y_qlow']).mean()
                        coverages.append(coverage)
                        widths.append(width)
                if coverages and widths:
                    results.append({
                        'tmp': tmp,
                        'model': model,
                        'dimension': dimension,
                        'interval_width_mean': sum(widths) / len(widths),
                        'interval_width_std': pd.Series(widths).std(),
                        'coverage_rate_mean': sum(coverages) / len(coverages),
                        'coverage_rate_std': pd.Series(coverages).std(),
                        'significant_test': binomtest(sum(c >= 0.9 for c in coverages), len(coverages), 0.9, alternative='two-sided').pvalue
                    })

results_df = pd.DataFrame(results)
print(results_df)