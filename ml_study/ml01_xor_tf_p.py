import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터 xor: 다르면 1 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=2, 
                activation='sigmoid')) #0이냐 1이냐 sigmoid. 2차원: inputdim  #sklearn의 Perceptron()과 동일

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가 예측
loss, acc = model.evaluate(x_data, y_data)
y_predict = model.predict(x_data)

print(x_data, '의 예측결과: ', y_predict)
print('모델의 loss : ', loss)
print('acc : ', acc)


# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과:  [[0.50843114]
#  [0.41797954]
#  [0.51518977]
#  [0.42457435]]
# 모델의 loss :  0.6995851993560791
# acc :  0.5
