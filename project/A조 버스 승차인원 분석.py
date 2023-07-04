import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
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

bus_data = preprocess_data(train_datasets)
bus_data = bus_data[~bus_data['date'].isin(['2019-09-01', '2019-09-07', '2019-09-08', '2019-09-12',
                                                           '2019-09-13', '2019-09-14', '2019-09-15', '2019-09-21',
                                                           '2019-09-22', '2019-09-28', '2019-09-29'])]

# 1시간 단위 데이터를 y데이터와 같이 2시간 단위로
bus_data['68a'] = bus_data['6~7_ride'] + bus_data['7~8_ride']
bus_data['810a'] = bus_data['8~9_ride'] + bus_data['9~10_ride']
bus_data['1012a'] = bus_data['10~11_ride'] + bus_data['11~12_ride']

bus_data['68b'] = bus_data['6~7_takeoff'] + bus_data['7~8_takeoff']
bus_data['810b'] = bus_data['8~9_takeoff'] + bus_data['9~10_takeoff']
bus_data['1012b'] = bus_data['10~11_takeoff'] + bus_data['11~12_takeoff']

# k-fold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

x = bus_data[['date', 'bus_route_id', 'station_code', 'station_name',
                   'latitude', 'longitude', '68a', '810a', '1012a', '68b', '810b', '1012b']]
y = bus_data['18~20_ride']

# 라벨인코딩
label_encoders = {} 
for col in ['date', 'station_name']:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le


print("\nAfter Label Encoding:")
print(x)

print("\nMissing values in train_datasets:")
print(pd.isna(bus_data))

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=28
)


# 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


##### 2. 모델
model = LGBMRegressor(colsample_bytree=0.5, n_estimators=824, num_leaves=127,
              random_state=28, reg_lambda=0.16, subsample=0.5)



##### 3. 교차 검증을 통한 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('cv r2:', scores.mean())

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('pred r2:', r2)



# 테스트 데이터 전처리

# id, in_out 컬럼제외, 휴일 데이터 제외
def preprocess_data(test_datasets):
    test_datasets = test_datasets.drop(['id', 'in_out'], axis=1)
    return test_datasets

bustest_data = preprocess_data(test_datasets)


# 1시간 단위 데이터를 y데이터와 같이 2시간 단위로
bustest_data['68a'] = bustest_data['6~7_ride'] + bustest_data['7~8_ride']
bustest_data['810a'] = bustest_data['8~9_ride'] + bustest_data['9~10_ride']
bustest_data['1012a'] = bustest_data['10~11_ride'] + bustest_data['11~12_ride']

bustest_data['68b'] = bustest_data['6~7_takeoff'] + bustest_data['7~8_takeoff']
bustest_data['810b'] = bustest_data['8~9_takeoff'] + bustest_data['9~10_takeoff']
bustest_data['1012b'] = bustest_data['10~11_takeoff'] + bustest_data['11~12_takeoff']

x1 = bustest_data[['date', 'bus_route_id', 'station_code', 'station_name',
                   'latitude', 'longitude', '68a', '810a', '1012a', '68b', '810b', '1012b']]



# 라벨인코딩
label_encoders = {} 
for col in ['date', 'station_name']:
    le = LabelEncoder()
    x1.loc[:, col] = le.fit_transform(x1[col])
    label_encoders[col] = le
    

print("\nAfter Label Encoding:")
print(x1.head())


# 스케일링
scaler = StandardScaler()
x1 = scaler.fit_transform(x1)


# 테스트 데이터 예측
test_y_result = model.predict(x1)
print(test_y_result)

# 테스트 결과 출력
bustest_data['18~20_ride'] = test_y_result
print(bustest_data.head(30))
