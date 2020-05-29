# Runs Laso and Ridge Regressions on the 30 columns I picked from univariate regression
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from ModelScore import model_score, model_features

# Global Parameters
use_dum = False
use_log = True
scale = True

# import the data
if use_dum:
    house = pd.read_csv('../derivedData/train_cleaned.csv', index_col='Id')
else:
    house = pd.read_csv('../derivedData/train_NotDum.csv', index_col='Id')
house['logSalePrice'] = np.log(house['SalePrice'])

if not(use_dum):
    # Use MV encoding on nominals
    cols_to_enc = house.columns[house.dtypes == 'object']
    for col in cols_to_enc:
        if use_log:
            gp = house.groupby(col)['logSalePrice'].mean()
        else:
            gp = house.groupby(col)['SalePrice'].mean()
        house[col] = house[col].apply(lambda x: gp[x])

# Create train and test sets
X = house.drop(['SalePrice', 'logSalePrice'],axis=1)
if use_log:
    y = house['logSalePrice']
else:
    y = house['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

if (scale):
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

# Initialise Lasso model
lasso = Lasso()

lasso_params = {'alpha': .02} # I actually overrode grid, which wanted 0.0001 and used pretty much all the columns
if lasso_params == {}:
    # Tune hyperparameters: takes about 3 minutes run time
    grid_para_lasso = [{'alpha': [1e-4, 1e-2, 0.1, 1, 2, 10, 100, 500, 750, 1000]}]
    grid_search_lasso = GridSearchCV(lasso, grid_para_lasso, cv=5, n_jobs=-1)
    start_t = time.time()
    grid_search_lasso.fit(X_train, y_train)
    end_t = time.time()
    print(f'Time taken: {end_t - start_t}')
    print('Best parameters: '+ str(grid_search_lasso.best_params_))
    lasso_final = grid_search_lasso.best_estimator_
else:
    lasso_final = lasso.set_params(**lasso_params).fit(X_train, y_train)

print(f'Lasso train score {lasso_final.score(X_train, y_train):.02%}')
print(f'Lasso test score {lasso_final.score(X_test, y_test):.02%}')
print(f'Lasso cols used {sum(lasso_final.coef_ != 0)}')

# Calculate feature importance
def t_stat(reg, X, y):
    col_bool = reg.coef_ != 0
    X_s = X.loc[:, col_bool]
    sse = np.sum((reg.predict(X) - y) ** 2, axis=0) / float(X_s.shape[0] - X_s.shape[1])
    try:
        xTx_inv = np.linalg.inv(X_s.T @ X_s)
        if min(np.diag(xTx_inv)) <= 0:
            t = np.full([1, X.shape[1]], np.nan)
        else:
            se_s = np.sqrt(np.diagonal(sse * xTx_inv))
            i = 0
            t = np.zeros([1, X.shape[1]])
            for j, b in enumerate(col_bool):
                if not(b): # i.e. reg.coef is zero
                    t[0, j] = 0
                else:
                    t[0, j] = reg.coef_[j] / se_s[i]
                    i += 1
    except:
        t = np.full([1, X.shape[1]], np.nan)
    return t[0]

if (scale):
    lasso_feature_imp = pd.Series(abs(lasso_final.coef_), index=X.columns).sort_values(ascending=False)
else:
    lasso_feature_imp = pd.Series(abs(t_stat(lasso_final, X_train, y_train)), index=X.columns).sort_values(ascending=False)
    
# Initialise Ridge model
ridge = Ridge()

ridge_params = {'alpha': 100}
if ridge_params == {}:
    # Tune hyperparameters: takes about 3 minutes run time
    grid_para_ridge = [{'alpha': [1e-4, 1e-2, 0.1, 1, 2, 10, 100, 500, 750, 1000]}]
    grid_search_ridge = GridSearchCV(ridge, grid_para_ridge, cv=5, n_jobs=-1)
    start_t = time.time()
    grid_search_ridge.fit(X_train, y_train)
    end_t = time.time()
    print(f'Time taken: {end_t - start_t}')
    print('Best parameters: '+ str(grid_search_ridge.best_params_))
    ridge_final = grid_search_ridge.best_estimator_
else:
    ridge_final = ridge.set_params(**ridge_params).fit(X_train, y_train)

print(f'Ridge train score {ridge_final.score(X_train, y_train):.02%}')
print(f'Ridge test score {ridge_final.score(X_test, y_test):.02%}')
print(f'Ridge cols used {sum(ridge_final.coef_ != 0)}')

if (scale):
    ridge_feature_imp = pd.Series(abs(ridge_final.coef_), index=X.columns).sort_values(ascending=False)
else:
    ridge_feature_imp = pd.Series(abs(t_stat(ridge_final, X_train, y_train)), index=X.columns).sort_values(ascending=False)

model_score(lasso_final, X_test, y_test, saves=False)
model_score(ridge_final, X_test, y_test, saves=False)

model_features(lasso_final, lasso_feature_imp.index, lasso_feature_imp, saves=False)
model_features(ridge_final, ridge_feature_imp.index, ridge_feature_imp, saves=False)