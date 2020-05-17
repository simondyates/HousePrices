import pandas as pd
dataset = 'test'
house = pd.read_csv('./' + dataset + '.csv')

# The data description identifies the following features as using NA to represent none
has_na =  ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
         'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
         'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
         'Fence', 'MiscFeature']
# These are all categorical with string values.  Let's replace with 'None'
for col in has_na:
    house.loc[house[col].isna(), col] = 'None'

print('Before: {0}'.format(house.columns[house.isna().any()]))
#Before: Index(['MSZoning', 'LotFrontage', 'Utilities', 'Exterior1st', 'Exterior2nd',
#       'MasVnrType', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
#       'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual',
#       'Functional', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'SaleType'],
#      dtype='object')
# Annoying: much worse than the training set

# Lot Frontage
# Same as with train
house.loc[house['LotFrontage'].isna(), 'LotFrontage'] = 0

# Mas* same as with train
house.loc[house['MasVnrArea'].isna(), 'MasVnrArea'] = 0
house.loc[house['MasVnrType'].isna(), 'MasVnrType'] = 'None'

# GarageYrBlt same as with train
house.loc[house['GarageYrBlt'].isna(), 'GarageYrBlt'] = house.loc[house['GarageYrBlt'].isna(), 'YearBuilt']

miss_cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
       'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageCars', 'GarageArea',
       'SaleType']

# Fortunately it turns out we're only missing very few entries for each of these
# Let's replace with the modal value
for col in miss_cols:
    miss_rows = house[col].isna()
    mode = house.loc[miss_rows==False, col].value_counts().index[0]
    print('{0}: Missing {1}, modal value {2}'.format(col, miss_rows.sum(), mode))
    house.loc[miss_rows, col] = mode

print('After: {0}'.format(house.columns[house.isna().any()]))
house.to_csv('./' + dataset + '_cleaned.csv', index=False)