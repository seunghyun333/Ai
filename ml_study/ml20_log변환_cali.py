#log 변환
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=[datasets.feature_names])
print(df)
print(df.head())
print(df['Population']) #다른데이터에 비해 숫자가 너무 큼 다른건 다 2자리 이하인데 이건 4자리

df['Population'] = np.log1p(df['Population'])
print(df['Population'].head())

'''
  Population
0   5.777652
1   7.784057
2   6.208590
3   6.326149
4   6.338594
'''