# Runs Laso and Ridge Regressions on the 30 columns I picked from univariate regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold

# import the data
house = pd.read_csv('./train_meanEnc.csv', index_col='Id')
y = house['SalePrice'].to_numpy().reshape(-1, 1)
# Box-Cox transform the y
y = np.log(y)
X = house.drop('SalePrice', axis=1)

# select columns
top30 = ['OverallQual', 'Neighborhood', 'GrLivArea', 'ExterQual', 'BsmtQual',
         'KitchenQual', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'FireplaceQu',
         'YearBuilt', 'YearRemodAdd', 'Foundation', 'MSSubClass', 'MasVnrArea',
         'HeatingQC', 'SaleType', 'SaleCondition', 'OverallCond', 'MSZoning',
         'WoodDeckSF', 'OpenPorchSF', 'HouseStyle', 'HalfBath', 'LotShape',
         'LotArea', 'CentralAir', 'Electrical', 'RoofStyle', 'PavedDrive']
X = X[top30]

# split into test / train
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2, random_state=0)

# create pipeline of dummification and scaling
ss = StandardScaler()
ss.fit(Xtrain)
Xs_train = ss.transform(Xtrain)

# initialise Lasso, Ridge and GridSearchCV
lasso = Lasso()
ridge = Ridge()
params_lasso = [{'alpha':[1e-4,1e-2,0.1,1,2,10,100, 150, 200, 250]}]
params_ridge = [{'alpha':[1e-4,1e-2,0.1,1,2,10,100, 150, 200, 250]}]
# kf5 = KFold(n_splits=5, shuffle=False) # not used because this is the default anyway
grid_lasso = GridSearchCV(estimator=lasso, param_grid=params_lasso) #, cv = kf5)
grid_ridge = GridSearchCV(estimator=ridge, param_grid=params_ridge) #, cv = kf5)
grid_lasso.fit(Xs_train, ytrain)
grid_ridge.fit(Xs_train, ytrain)

# pick winning model
lasso_score = grid_lasso.best_score_
ridge_score = grid_ridge.best_score_
lasso_alpha = grid_lasso.best_params_['alpha']
ridge_alpha = grid_ridge.best_params_['alpha']

print(f'In Sample Lasso: {lasso_score:.2%}')
print(f'In Sample Ridge: {ridge_score:.2%}')
print(f'Lasso alpha: {lasso_alpha:.4f}')
print(f'Ridge alpha: {ridge_alpha:.4f}')

# evaluate performance on OOS data
Xs_test = ss.transform(Xtest)
print( f'OOS Lasso {grid_lasso.best_estimator_.score(Xs_test,ytest):.2%}')
print( f'OOS Ridge {grid_ridge.best_estimator_.score(Xs_test, ytest):.2%}')

lasso_coefs = grid_lasso.best_estimator_.coef_
ridge_coefs = grid_ridge.best_estimator_.coef_

print(f'Lasso used {(lasso_coefs!=0).sum()} columns')
print(f'Ridge used {(ridge_coefs!=0).sum()} columns')