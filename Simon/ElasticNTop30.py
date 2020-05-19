# I've manually curated the results of the univariate regressions
# by removing variables that just seem like they're measuring the same thing
# i.e. only pick one variable about the garage
# This regression uses the Top 30 univariate R^2 after this curation

import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
import numpy as np
import plotly.graph_objects as go

house = pd.read_csv('./train_meanEnc.csv', index_col='Id')
# Split target from attributes and normalise attribs
Y = house['SalePrice'].to_numpy().reshape(-1, 1)

top30 = ['OverallQual', 'Neighborhood', 'GrLivArea', 'ExterQual', 'BsmtQual',
         'KitchenQual', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'FireplaceQu',
         'YearBuilt', 'YearRemodAdd', 'Foundation', 'MSSubClass', 'MasVnrArea',
         'HeatingQC', 'SaleType', 'SaleCondition', 'OverallCond', 'MSZoning',
         'WoodDeckSF', 'OpenPorchSF', 'HouseStyle', 'HalfBath', 'LotShape',
         'LotArea', 'CentralAir', 'Electrical', 'RoofStyle', 'PavedDrive']

X = house[top30]
X = (X - X.mean(axis=0)) / X.std(axis=0)

regr = LinearRegression()
regr.fit(X, Y)

#print(regr.alpha_)
print(regr.intercept_)
print(regr.score(X, Y))