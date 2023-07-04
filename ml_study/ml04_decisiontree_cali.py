# 4.california svr 회귀분석 - SVR, LinearSVR 
# scaler 사이킷런에서 제공 어떤 걸 적용하느냐에 따라 값이 다름 

import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수 값 조정 

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #변수, 컬럼, 열 
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100, shuffle=True)

#Scaler 적용 => train과 test 둘 다 적용해야함 
scaler = StandardScaler()
scaler.fit(x_train)  #왜 train만 하는지 확인하기 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = DecisionTreeRegressor()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
result = model.score(x_test,y_test) 
y_predict = model.predict(x_test)
print('결과 r2 : ', result)

#SVR 결과 acc:  -0.01663695941103427
#Linear SVC 결과 acc:   -0.055781872981941705
#딥러닝: accuracy 결과 acc:   -1.8172916629018916
#                     r2 :  -1.8172916629018916
 
#sclaer MinMaxScaler 
# r2 :  0.6026608996666052

#scaler = StandardScaler
# r2 :  0.36326347111674373

# scaler = MaxAbsScaler()
# r2 :  0.5622406660414347

#scaler = RobustScaler()
#결과 r2 :  -0.7148101761327619

#decision tree- regressor
#Minmax: 결과 r2 :  0.6252904174582093
#Standard 결과 : 0.610199889411199
