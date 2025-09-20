import pandas as pd
import numpy as np
from scipy import stats

def coverage_and_width(low, up, y_test):
    width = up - low
    coverage = np.mean((low <= y_test) & (y_test <= up))
    return width.mean(), coverage.mean()

def calculate_performance(y_pred, y_test):
    if np.isnan(y_pred).any() or np.isnan(y_test).any():
        mask = ~np.isnan(y_pred) & ~np.isnan(y_test)
        y_pred = y_pred[mask]
        y_test = y_test[mask]
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test))

    rho = stats.spearmanr(y_pred, y_test)[0]
    tau = stats.kendalltau(y_pred, y_test)[0]
    pcc = stats.pearsonr(y_pred, y_test)[0]
    return mse, rmse, mae, rho, tau, pcc

dimension = 'drop'
file_path = f'loo_reprompt_socreval_{dimension}.csv'  
df = pd.read_csv(file_path)
df = df.iloc[:,:-1]

low = df.iloc[:,1].to_numpy().astype(np.float32)
up = df.iloc[:,2].to_numpy().astype(np.float32)
y_test = df.iloc[:,-1].to_numpy().astype(np.float32)
y_pred = df.iloc[:,4].to_numpy().astype(np.float32)
init_score_weight = [float(item.strip('[]')) for item in df.iloc[:,3]]

midpoint = (low+up)/2

width_init, coverage_init = coverage_and_width(low, up, y_test)

print(width_init)
print(coverage_init)

print("midpoints")
mse, rmse, mae, rho, tau, pcc = calculate_performance(midpoint, y_test)
print(mse)
print(rmse)
print(mae)
print(rho)
print(tau)
print(pcc)

print("init_weighted")
mse, rmse, mae, rho, tau, pcc = calculate_performance(init_score_weight, y_test)
print(mse)
print(rmse)
print(mae)
print(rho)
print(tau)
print(pcc)


print("reprompt_weighted")
mse, rmse, mae, rho, tau, pcc = calculate_performance(y_pred, y_test)
print(mse)
print(rmse)
print(mae)
print(rho)
print(tau)
print(pcc)