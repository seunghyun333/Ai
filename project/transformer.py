# 데이터 처리 및 전처리
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# 모델 및 학습
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score
# 모델 구현
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

import time

###### 1. 데이터
path = './_data2/'
train_datasets = pd.read_csv(path + 'train.csv')
test_datasets = pd.read_csv(path + 'test.csv')

print(train_datasets.columns)
print('\n-----------------\n')
print(train_datasets.head(7))

# id, in_out 컬럼제외, 휴일 데이터 제외
def preprocess_data(train_datasets):
    train_datasets = train_datasets.drop(['id', 'in_out'], axis=1)
    return train_datasets

train = preprocess_data(train_datasets)
train = train[~train['date'].isin(['2019-09-01', '2019-09-07', '2019-09-08', '2019-09-12',
                                                           '2019-09-13', '2019-09-14', '2019-09-15', '2019-09-21',
                                                           '2019-09-22', '2019-09-28', '2019-09-29'])]

# 1시간 단위 데이터를 y데이터와 같이 2시간 단위로
train['68a'] = train['6~7_ride'] + train['7~8_ride']
train['810a'] = train['8~9_ride'] + train['9~10_ride']
train['1012a'] = train['10~11_ride'] + train['11~12_ride']

train['68b'] = train['6~7_takeoff'] + train['7~8_takeoff']
train['810b'] = train['8~9_takeoff'] + train['9~10_takeoff']
train['1012b'] = train['10~11_takeoff'] + train['11~12_takeoff']

# k-fold
n_splits = 17
random_state = 342
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#cv r2: 0.6936142584412637

x = train[['date', 'bus_route_id', 'station_code', 'station_name',
                   'latitude', 'longitude', '68a', '810a', '1012a', '68b', '810b', '1012b']]
y = train['18~20_ride']

# 라벨인코딩
label_encoders = {}
for col in ['date', 'station_name']:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le

# print("Before Label Encoding:")
# print("x shape:", x.shape)
# print("y shape:", y.shape)

print("\nAfter Label Encoding:")
print(x)

print("\nMissing values in train_datasets:")
print(pd.isna(train))

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=55
)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
'''
# 스케일링
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import VotingRegressor

# 개별 모델 생성
model1 = XGBRegressor()
model2 = CatBoostRegressor()
model3 = LGBMRegressor()
model4 = RandomForestRegressor()

# 앙상블 모델 생성
ensemble_model = VotingRegressor([('xgb', model1), ('catboost', model2), ('lgbm', model3), ('rf', model4)])

# 교차 검증을 통한 평가
scores = cross_val_score(ensemble_model, x_train, y_train, cv=kfold, scoring='r2')

# 앙상블 모델 학습
start_time = time.time ()
ensemble_model.fit(x_train, y_train)
end_time = time.time () - start_time

# 앙상블 모델 예측
y_predict = ensemble_model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('cv r2:', scores.mean())
print('r2 score:', r2)
print('걸린시간 : ', end_time)
'''
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

# 스케일링
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer()                     # QuantileTransformer 는 지정된 분위수에 맞게 데이터를 변환함. 
                                                # 기본 분위수는 1,000개이며, n_quantiles 매개변수에서 변경할 수 있음
ptf1 = PowerTransformer(method='yeo-johnson')   # 'yeo-johnson', 양수 및 음수 값으로 작동
ptf2 = PowerTransformer(method='box-cox')       # 'box-cox', 양수 값에서만 작동

scalers = [sts, mms, mas, rbs, qtf, ptf1, ptf2]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    scale_name = scaler.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(scale_name, result), )
    