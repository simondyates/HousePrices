import pandas as pd
dataset = 'train'
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
# Before: Index(['LotFrontage', 'MasVnrType', 'MasVnrArea', 'Electrical', 'GarageYrBlt'], dtype='object')

# Lot Frontage
# It's likely that LotFrontage is NaN when there is none.
# The R dataset has a comparable proportion of zeros and no Nans
house.loc[house['LotFrontage'].isna(), 'LotFrontage'] = 0

# MasVrnArea is na when MasVnrType is also NA.  Probably best to zero the float and 'None' the string
house.loc[house['MasVnrArea'].isna(), 'MasVnrArea'] = 0
house.loc[house['MasVnrType'].isna(), 'MasVnrType'] = 'None'

# There's only one missing value in Electrical
print(house['Electrical'].value_counts())
#SBrkr    1334
#FuseA      94
#FuseF      27
#FuseP       3
#Mix         1
#Name: Electrical, dtype: int64

# Let's replace that with the modal value
house.loc[house['Electrical'].isna(), 'Electrical'] = 'SBrkr'

# GarageYrBlt has NA when GarageType = 'None'
print(house.loc[house['GarageYrBlt'].isna(), 'GarageType'].value_counts())
#None    81
#Name: GarageType, dtype: int64

# We could encode that as 0 but this will lead to a wide data range (0s and 1985 for example)
# I think it might be better to define it as equal to house built date to avoid this
house.loc[house['GarageYrBlt'].isna(), 'GarageYrBlt'] = house.loc[house['GarageYrBlt'].isna(), 'YearBuilt']

print('After: {0}'.format(house.columns[house.isna().any()]))
house.to_csv('./' + dataset + '_cleaned.csv', index=False)