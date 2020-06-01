from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def linReg(type='None', use_log=True, scale=True, use_dum=False):
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

    if type.lower() == 'ridge':
        reg = Ridge(max_iter=1e6)
        params = {
            'alpha': np.linspace(50, 150, 50),
            'solver': ['auto', 'cholesky', 'svd', 'lsqr', 'saga']
        }
    elif type.lower() == 'lasso':
        reg = Lasso(max_iter=1e6)
        params = {
            'alpha': [.01],
            'selection': ['cyclic', 'random']
        }
    elif type.lower() in ['enet', 'elasticnet', 'elastic']:
        reg = ElasticNet(max_iter=1e8)
        params = {
            'alpha': np.linspace(1e-6, 400, 20),
            'l1_ratio': np.linspace(0.011, 1, 20),
            'selection': ['cyclic', 'random']
        }
    else:
        reg = LinearRegression()
        params = {}

    grid_search_reg = GridSearchCV(reg, params, cv=4)

    grid_search_reg.fit(X_train, y_train)

    return grid_search_reg