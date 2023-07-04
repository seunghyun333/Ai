# [실습] svm 모델과 나의 tf keras 모델 성능비교하기
# 1.iris 다중분류 SVC, LinveraSVC
# 2.cancer 이진분류 SVC, LinveraSVC
# 3.wine    다중분류 SVC, LinveraSVC
# 4.california svr 회귀분석 - SVR, LinearSVR 

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수 값 조정 

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100)

# sclaer 적용
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = SVC()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
result = model.score(x_test,y_test) 
print('결과 acc:  ', result)

#SVC 결과 acc:  0.5555555555555556
#Linear SVC 결과 acc:   0.6851851851851852
#딥러닝: accuracy :  0.7777777910232544

#Scaler 적용 
#Standard : 0.9814814814814815
#Minmax : 0.9814814814814815
#MaxAbs: 0.9814814814814815
#Robuster:   0.9814814814814815