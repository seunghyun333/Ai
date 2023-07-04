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
path = './'
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

# print("Before Label Encoding:")
# print("x shape:", x.shape)
# print("y shape:", y.shape)

print("\nAfter Label Encoding:")
print(x)

print("\nMissing values in train_datasets:")
print(pd.isna(bus_data))

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=55
)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#parameters
# param =  {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'num_leaves': [20, 30, 40]
# }
param =  {'num_leaves': [31, 127],  
              'learning_rate': [0.01, 0.1, 0.5],  
              'n_estimators': [100, 500, 1000], 
              'subsample': [0.5, 0.8, 1.0],  
              'colsample_bytree': [0.5, 0.8, 1.0],  
              'reg_alpha': [0.0, 0.1, 0.5],  
              'reg_lambda': [0.0, 0.1, 0.5],  
              'random_state': [42]
             }
        


##### 2. 모델
lgbm_model = LGBMRegressor()
model = GridSearchCV(lgbm_model, param, refit=True, cv=kfold, verbose=1, n_jobs=-1)


##### 3. 교차 검증을 통한 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('cv r2:', scores.mean())

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time
print("최적의 파라미터 : ", model.best_params_)
print("최적의 매개변수 : ", model.best_estimator_)
print("best_score : ", model.best_score_)
print("model_score : ", model.score(x_test, y_test))
print("걸린 시간 : ", end_time, "초")
# 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'num_leaves': 40}
# 최적의 매개변수 :  LGBMRegressor(max_depth=7, n_estimators=200, num_leaves=40)
# best_score :  0.6730006327052649
# model_score :  0.7202654133732147
# 걸린 시간 :  46.52400016784668 초
# cv pred r2: 0.7202654133732147

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('cv pred r2:', r2)

# cv r2: 0.7265542610196387
# cv pred r2: 0.7661302863150565