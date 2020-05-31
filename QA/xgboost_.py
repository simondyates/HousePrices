import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from HousePrices.Dan.ModelScore import model_score


def xgboost(booster = 'gblinear', use_log = True, scale = True, use_dum = False):
    # preparing the data
    if use_dum:
        data = pd.read_csv('../derivedData/train_cleaned.csv', index_col='Id')
    else:
        data = pd.read_csv('../derivedData/train_NotDum.csv', index_col='Id')

    data['logSalePrice'] = np.log(data['SalePrice'])

    if not use_dum:
        cols_to_enc = data.columns[data.dtypes == 'object']
        for col in cols_to_enc:
            if use_log:
                gp = data.groupby(col)['logSalePrice'].mean()
            else:
                gp = data.groupby(col)['SalePrice'].mean()
            data[col] = data[col].apply(lambda x: gp[x])

    X = data.drop(['SalePrice', 'logSalePrice'], axis=1)
    if not use_log:
        y = data['SalePrice']
    else:
        y = data['logSalePrice']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if scale:
        ss = StandardScaler()
        ss.fit(X_train)
        X_train = pd.DataFrame(ss.transform(X_train))
        X_test = pd.DataFrame(ss.transform(X_test))

    if booster == 'gblinear':
        xgb_param = {
            'alpha': [0],
            'lambda': np.linspace(0, .2, 200)
        }
    elif booster == 'gbtree':
        xgb_param = {
            'max_depth': [2, 3],
            'min_child_weight': np.linspace(5, 15, 20),
            'lambda': np.linspace(1, 10, 20),
            'alpha': [0]
        }
    elif booster == 'dart':
        xgb_param = {
            'max_depth': [2],
            'min_child_weight': np.linspace(10, 15, 6),
            'lambda': np.linspace(0, 2, 4),
            'alpha': [0],
            'sample_type': ['uniform'],
            'normalize_type': ['tree'],
            'rate_drop': np.linspace(.5, 1, 8),
            'skip_drop': np.linspace(.5, 1, 10)
        }

    xgboost = xgb.XGBRegressor(booster=booster)

    grid_search_xgb = GridSearchCV(xgboost, xgb_param, cv=4)

    grid_search_xgb.fit(X_train, y_train)

    return grid_search_xgb
    # model_score(grid_search_xgb.best_estimator_, X_test, y_test, saves=True)
