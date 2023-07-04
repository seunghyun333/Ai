# 4.california svr 회귀분석 - SVR, LinearSVR 
# scaler 사이킷런에서 제공 어떤 걸 적용하느냐에 따라 값이 다름 

import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수 값 조정 

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #변수, 컬럼, 열 
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100, shuffle=True)

#kfold
n_splits = 5
random_state =42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#Scaler 적용 => train과 test 둘 다 적용해야함 
scaler = StandardScaler()
scaler.fit(x_train)  #왜 train만 하는지 확인하기 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = RandomForestRegressor()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
score = cross_val_score(model,
                        x_train, y_train,
                        cv=kfold)
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
r2 = r2_score(y_test, y_predict)

print("cross validation acc: ",score)
#[0.79243608 0.81931629 0.79251683 0.79680168 0.78992854]
print('cv predict : ', y_predict)
#cv predict :  [2.3762601 1.2960601 1.71751   ... 0.99277   1.47249   4.9632895]
print('r2 score :', r2)
#r2 score : 0.7736576724548081



#SVR 결과 acc:  -0.01663695941103427
#Linear SVC 결과 acc:   -0.055781872981941705
#딥러닝: accuracy 결과 acc:   -1.8172916629018916
#                     r2 :  -1.8172916629018916
 
 
#sclaer MinMaxScaler r2 :  0.6026608996666052
#scaler = StandardScaler  r2 :  0.36326347111674373
# scaler = MaxAbsScaler()  r2 :  0.5622406660414347
#scaler = RobustScaler()  r2 :  -0.7148101761327619

#decision tree- regressor
#Minmax: 결과 r2 :  0.6252904174582093
#Standard 결과 : 0.610199889411199

#앙상블
#standard : 0.8134721940607257
