import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

# 1. 데이터
path = './_data2/'
datasets = pd.read_csv (path + 'train.csv')

  # 입력 변수들의 데이터프레임
filtered_data = datasets[datasets['date'].isin(['2019-09-01', '2019-09-07','2019-09-07','2019-09-08','2019-09-12','2019-09-13',
                                                '2019-09-14','2019-09-15','2019-09-21','2019-09-22','2019-09-28','2019-09-29']) == False]

# 필터링된 데이터를 x, y로 분할
x = filtered_data[['date', 'bus_route_id', 'in_out', 'station_code', 'station_name',
       'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
       '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
       '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']]
y = filtered_data[['18~20_ride']] 

# print (x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=42, shuffle=True)

from sklearn.preprocessing import LabelEncoder

# LabelEncoder를 적용할 열을 선택합니다.
cols_to_encode = ['date', 'in_out', 'station_name']

# LabelEncoder 객체를 생성하고 열을 순회하며 변환합니다.
label_encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    x.loc[:, col] = le.fit_transform(x[col])
    label_encoders[col] = le


null_counts = x.isnull().sum()
# print(null_counts)

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

# 훈련 세트와 테스트 세트로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=777)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델(conda에서 catboost 설치)
from catboost import CatBoostRegressor
model = CatBoostRegressor()

# x_train과 y_train을 NumPy 배열로 변환
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

model.fit(x_train,y_train)

from sklearn.model_selection import KFold
#kfold train,test 셋 나눌 때 아무데나 넣어도 상관없음. kfold 선언이라서 
n_splits = 5 # 11이면 홀수 단위로 자름 10개가 train, 1개가 test
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) 

from sklearn.model_selection import cross_val_score, cross_val_predict

score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) #cv: cross validation

y_predict = cross_val_predict(model, 
                              x_test, y_test,
                              cv=kfold)
r2 = r2_score(y_test, y_predict)

print('r2 :', r2)
