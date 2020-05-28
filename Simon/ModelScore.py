import pandas as pd
import numpy as np
import os
from datetime import datetime

def model_score(model, X_test, y_test, prints=True, saves=True):
    model_name = type(model).__name__
    pred_y_test = model.predict(X_test)
    if max(y_test) < 15:
        pred_y_test = np.exp(pred_y_test)
        y_test = np.exp(y_test)
    n = len(y_test)
    p = X_test.shape[1]
    bias = (pred_y_test - y_test).mean()
    maxDev = abs(pred_y_test - y_test).max()
    meanAbsDev = abs(pred_y_test - y_test).mean()
    MSE = ((pred_y_test - y_test)**2).mean()
    MTE = ((y_test - y_test.mean())**2).mean()
    MSM = ((pred_y_test - y_test.mean())**2).mean()
    R2 = 1 - MSE/MTE
    Adj_R2 = 1 - ( (1 - R2)*(n - 1) / ( n-p-1) )
    skew = ((pred_y_test - y_test)**3).mean() / MSE**1.5
    kurt = ((pred_y_test - y_test)**4).mean() / MSE**2 - 3
    AIC = n * np.log(MSE) + 2 * (p+1)
    F = (MSM / MSE) * ((p-1)/(n-p))
    user_name = os.environ.get('USER')
    dt_stamp = datetime.now().strftime('%Y-%m-%d %H-%M')

    data = [model_name, bias, maxDev, meanAbsDev, MSE**0.5, R2, Adj_R2, skew, kurt, AIC, F, user_name, dt_stamp]
    idx = ['Model', 'Bias', 'MaxDev', 'MeanAbsDev', 'RMSE', 'R2', 'Adj_R2', 'Skew', 'Kurt', 'AIC', 'F', 'User', 'Date']
    results = pd.Series(data, index=idx)
    if (prints):
        print('-' * len(model_name))
        print(model_name)
        print('-'*len(model_name))
        print(f'Bias: {bias:,.0f}')
        print(f'Max Dev: {maxDev:,.0f}')
        print(f'Mean Abs Dev: {meanAbsDev:,.0f}')
        print(f'RMSE: {MSE**0.5:,.0f}')
        print(f'R^2: {R2:.2%}')
        print(f'Adj R^2: {Adj_R2:.2%}')
        print(f'Resid skew: {skew:.2f}')
        print(f'Resid kurt: {kurt:.2f}')
        print(f'AIC: {AIC:,.2f}')
        print(f'F: {F:.2f}')
        print('-' * len(model_name))
    if (saves):
        results.to_csv(f'../results/{model_name[:10]} {dt_stamp}.csv')
