# Runs Random Forest and Gradient Boosting Regressions
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from ModelScore import model_score, model_features

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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Initialise RF model
randomForest = ensemble.RandomForestRegressor(random_state=0)

rF_params = {'max_depth': 30, 'max_features': 30, 'max_samples': None, 'min_samples_leaf': 1, 'n_estimators': 500}
if rF_params == {}:
    # Tune hyperparameters: takes about 3 minutes run time
    grid_para_forest = [{
        'n_estimators': [50, 100, 500],
        'min_samples_leaf': [1, 10, 30],
        'max_depth': [10, 30, None],
        'max_features': [10, 30, 'auto'],
        'max_samples': [.25, .5, .75, None]
        }]
    grid_search_forest = GridSearchCV(randomForest, grid_para_forest, cv=5, n_jobs=-1)
    start_t = time.time()
    grid_search_forest.fit(X_train, y_train)
    end_t = time.time()
    print(f'Time taken: {end_t - start_t}')
    print('Best parameters: '+ str(grid_search_forest.best_params_))
    rF_final = grid_search_forest.best_estimator_
else:
    rF_final = randomForest.set_params(**rF_params).fit(X_train, y_train)

print(f'RF train score {rF_final.score(X_train, y_train):.02%}')
print(f'RF test score {rF_final.score(X_test, y_test):.02%}')
rF_feature_imp = pd.Series(rF_final.feature_importances_, index=X.columns).sort_values(ascending=False)

# Initialise gradient boost model
gradBoost = ensemble.GradientBoostingRegressor(random_state=0)

gB_params = {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 500, 'subsample': 0.75}
if gB_params == {}:
    # Tune hyperparameters: takes about 1 minute run time
    grid_para_forest = [{
        'n_estimators': [50, 100, 500],
        'learning_rate': [.1, .3, .5],
        'max_depth': [2, 3, 5],
        'subsample': [.5, .75, 1]
        }]
    grid_search_boost = GridSearchCV(gradBoost, grid_para_forest, cv=5, n_jobs=-1)
    start_t = time.time()
    grid_search_boost.fit(X_train, y_train)
    end_t = time.time()

    print(f'Time taken: {end_t - start_t}')
    print('Best parameters: '+ str(grid_search_boost.best_params_))
    gB_final = grid_search_boost.best_estimator_
else:
    gB_final = gradBoost.set_params(**gB_params).fit(X_train, y_train)

print(f'G Boost train score {gB_final.score(X_train, y_train):.02%}')
print(f'G Boost test score {gB_final.score(X_test, y_test):.02%}')
gB_feature_imp = pd.Series(gB_final.feature_importances_, index=X.columns).sort_values(ascending=False)

model_score(rF_final, X_test, y_test, saves=False)
model_score(gB_final, X_test, y_test, saves=False)

model_features(rF_final, X_test, y_test, X.columns, saves=False)
model_features(gB_final, X_test, y_test, X.columns, saves=False)