# Runs Support Vector Regression
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Global Parameters
use_dum = False
use_log = True

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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Initialise RF model
svr = svm.SVR()

svr_params = {'C': 100, 'epsilon': 0.1, 'gamma': 'scale'}
if svr_params == {}:
    # Tune hyperparameters: quick unless you test for poly when it seems never to come back
    grid_para_svr = [{
        'gamma': ['scale', 100, 50], # big gamma means low var so precise
        'C':[500, 100, 10], # big value means don't let errors happen
        'epsilon': [.1, .01, .001] # big value means don't penalize a wide range around the boundary
        }]
    grid_search_svr = GridSearchCV(svr, grid_para_svr, cv=5, n_jobs=-1)
    start_t = time.time()
    grid_search_svr.fit(X_train, y_train)
    end_t = time.time()
    print(f'Time taken: {end_t - start_t}')
    print('Best parameters: '+ str(grid_search_svr.best_params_))
    svr_final = grid_search_svr.best_estimator_
else:
    svr_final = svr.set_params(**svr_params).fit(X_train, y_train)


print(f'SVR train score {svr_final.score(X_train, y_train):.02%}')
print(f'SVR test score {svr_final.score(X_test, y_test):.02%}')