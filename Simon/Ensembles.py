# Runs Random Forest and Gradient Boosting Regressions
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

# import the non-dummified data
house = pd.read_csv('../derivedData/train_NotDum.csv', index_col='Id')
house['logSalePrice'] = np.log(house['SalePrice'])
use_log = False

# Use MV encoding on nominals to avoid column bloat
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

# Initialise model
randomForest = ensemble.RandomForestRegressor(random_state=0)
#bagging      = ensemble.BaggingClassifier()
#randomForest.fit(X_train, y_train)
#print(f'RF train score {randomForest.score(X_train, y_train):.02%}')
#print(f'RF test score {randomForest.score(X_test, y_test):.02%}')
#feature_importance = randomForest.feature_importances_
#s = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)

grid_para_forest = [{
    #'n_estimators': [50, 100, 500],
    #'min_samples_leaf': [1, 10, 30],
    #'max_depth': [3, 10, 30],
    #'max_features': [10, 30, 50],
    'max_samples': [.25, .5, .75, 1]
    }]
grid_search_forest = GridSearchCV(randomForest, grid_para_forest, cv=5, n_jobs=-1)
start_t = time.time()
grid_search_forest.fit(X_train, y_train)
end_t = time.time()
print(end_t - start_t)
