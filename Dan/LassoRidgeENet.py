from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from HousePrices.Dan.ModelScore import model_score

# Global parameters
logprice = True
model = 'enet'
scale = True
use_dum = False

# preparing the data
if use_dum:
    data = pd.read_csv('../derivedData/train_cleaned.csv', index_col='Id')
else:
    data = pd.read_csv('../derivedData/train_NotDum.csv', index_col='Id')

data['logSalePrice'] = np.log(data['SalePrice'])

if not use_dum:
    cols_to_enc = data.columns[data.dtypes == 'object']
    for col in cols_to_enc:
        if logprice:
            gp = data.groupby(col)['logSalePrice'].mean()
        else:
            gp = data.groupby(col)['SalePrice'].mean()
        data[col] = data[col].apply(lambda x: gp[x])

X = data.drop(['SalePrice', 'logSalePrice'], axis=1)
if not logprice:
    y = data['SalePrice']
else:
    y = data['logSalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if scale:
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = pd.DataFrame(ss.transform(X_train))
    X_test = pd.DataFrame(ss.transform(X_test))

if model.lower() == 'ridge':
    reg = Ridge(max_iter=1e6)
    params = {
        'alpha': np.linspace(50, 150, 100),
        'solver': ['auto', 'cholesky', 'svd', 'lsqr', 'saga']
    }
elif model.lower() == 'lasso':
    reg = Lasso(max_iter=1e6)
    params = {
        'alpha': np.linspace(1e-6, 1, 20),
        'selection': ['cyclic', 'random']
    }
elif model.lower() in ['enet', 'elasticnet', 'elastic']:
    reg = ElasticNet(max_iter=1e8)
    params = {
        'alpha': np.linspace(1e-6, 400, 20),
        'l1_ratio': np.linspace(0.011, 1, 20),
        'selection': ['cyclic', 'random']
    }

grid_search_reg = GridSearchCV(reg, params, cv=4)

start_t = time.time()
grid_search_reg.fit(X_train, y_train)
end_t = time.time()

print(f'Time taken: {end_t - start_t}')
print('Best parameters: ' + str(grid_search_reg.best_params_))

model_score(grid_search_reg.best_estimator_, X_test, y_test, saves=False)
