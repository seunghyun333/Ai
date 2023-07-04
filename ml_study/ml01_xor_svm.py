import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score

#1. 데이터 xor: 다르면 1 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델 
model = SVC()   #멀티레이어(MLP) 퍼셉트론  레이어가 쌓이면 접을 수 있음

#3. 훈련
model.fit(x_data, y_data)

#4. 평가 예측
result = model.score(x_data, y_data)
y_predict = model.predict(x_data)
acc = accuracy_score(y_data, y_predict)

print('모델의 score : ', result)
print(x_data, '의 예측결과 : ',y_predict)
print('acc : ',acc)

# 모델의 score :  1.0
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 0]
# acc :  1.0

