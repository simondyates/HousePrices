# This takes an aggressive approach of encoding all categorical variables
# by the mean value of the SalePrice

import pandas as pd

house = pd.read_csv('./train_cleaned.csv', index_col='Id')

noms = ['MSSubClass', 'MSZoning', 'Street', 'Alley',
       'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType',
       'MiscFeature', 'SaleType', 'SaleCondition']

ords = ['LotShape', 'LandContour', 'Utilities', 'LandSlope', 'OverallQual',
         'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
         'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
         'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
         'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

categors = noms + ords

house_enc = house.copy()
for col in categors:
    gp = house.groupby(col)['SalePrice'].mean()
    house_enc[col] = house_enc[col].apply(lambda x: gp[x])

house_enc.to_csv('./train_meanEnc.csv')