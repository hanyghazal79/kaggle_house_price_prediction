import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_cols = train.columns
train.info()
# Checking nan values
nans = train.isna().sum()
nans_todrop = nans[nans >= 0.2*len(train)]
cols_todrop = nans_todrop.index.tolist()
# deleting nan cols having >= 20% of data missed
train.drop(columns=['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)
train['SalePrice'].describe()
sn.displot(data=train['SalePrice']) # positively skewed distribution # Peakedness
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())



