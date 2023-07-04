# [실습] svm 모델과 나의 tf keras 모델 성능비교하기
# 1.iris 다중분류 SVC, LinveraSVC
# 2.cancer 이진분류 SVC, LinveraSVC
# 3.wine    다중분류 SVC, LinveraSVC
# 4.california svr 회귀분석 - SVR, LinearSVR 

import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(77) # weight 난수 값 조정 

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #변수, 컬럼, 열 
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100)

#print(x.shape, y.shape) #(20640, 8) (20640,)


#2. 모델
model = LinearSVR()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
result = model.score(x_test,y_test) 
y_predict = model.predict(x_test)
print('결과 acc:  ', result)
print('r2 : ', result)

#SVR 결과 acc:  -0.01663695941103427
#Linear SVC 결과 acc:   -0.055781872981941705
#딥러닝: accuracy :  결과 acc:   -1.8172916629018916
 #                   r2 :  -1.8172916629018916