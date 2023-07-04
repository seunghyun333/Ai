import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터 xor: 다르면 1 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#[실습] MLP 모델 구성하여 acc = 1.0 만들기
#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=2)) #MLP(multi layer perceptron)
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가 예측
loss, acc = model.evaluate(x_data, y_data)
y_predict = model.predict(x_data)

print(x_data, '의 예측결과: ', y_predict)
print('모델의 loss : ', loss)
print('acc : ', acc)


#멀티레이어 퍼셉트론
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과:  [[5.5717781e-05]
#  [9.9752456e-01]
#  [9.9756634e-01]
#  [1.4295453e-03]]
# 모델의 loss :  0.0016003483906388283
# acc :  1.0
