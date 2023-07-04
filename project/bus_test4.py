import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostRegressor

# 1. 데이터
path = './_data2/'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.columns)
print('\n-----------------\n')
print(datasets.head(7))

# 데이터 전처리 id, in_out 컬럼제외, 휴일 데이터 제외
def preprocess_data(datasets):
    datasets = datasets.drop(['id', 'in_out'], axis=1)
    return datasets

filtered_data = preprocess_data(datasets)
filtered_data = filtered_data[~filtered_data['date'].isin(['2019-09-01', '2019-09-07', '2019-09-08', '2019-09-12',
                                                           '2019-09-13', '2019-09-14', '2019-09-15', '2019-09-21',
                                                           '2019-09-22', '2019-09-28', '2019-09-29'])]

# k-fold
n_splits = 7
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

x = filtered_data[['date', 'bus_route_id', 'station_code', 'station_name',
                   'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
                   '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff',
                   '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff',
                   '11~12_takeoff']]
y = filtered_data['18~20_ride']

# 라벨인코딩
label_encoders = {}
for col in ['date', 'station_name']:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le

print("Before Label Encoding:")
print("x shape:", x.shape)
print("y shape:", y.shape)

print("\nAfter Label Encoding:")
print(x)

print("\nMissing values in datasets:")
print(pd.isna(filtered_data))

# 스케일링
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=True, random_state=777)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델
model = CatBoostRegressor()

# 교차 검증을 통한 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('cv r2:', scores.mean())

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('cv pred r2:', r2)

'''
# 특성 중요도 시각화
feature_importances = model.feature_importances_
n_features = len(feature_importances)

plt.barh(range(n_features), feature_importances, align='center')
plt.yticks(np.arange(n_features), x.columns)
plt.title('BUS Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.ylim(-1, n_features)

plt.show()

# 분포 그래프
plt.hist(y, bins=30)
plt.title('Distribution of Target Variable')
plt.xlabel('Target Variable (18~20_ride)')
plt.ylabel('Frequency')

plt.show()
'''