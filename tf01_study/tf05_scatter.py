#라이브러리 구성, 정리
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, train_size=0.7,random_state=100, shuffle=True    
)


#2.모델 구성
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss는', loss)

y_predict = model.predict(x)

##### 시각화 scatter
import matplotlib.pyplot as plt

plt.scatter(x, y) # 산점도 그리기, x.와 y값
plt.plot(x, y_predict, color='pink')
plt.show()

result = model.predict([21])
print('21의 예측값 : ', result)


