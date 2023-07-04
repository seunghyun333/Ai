#라이브러리 정리
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6 ]])  #리스트 안에 리스트 두개 2차원배열열
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)  #(2,10)  행과 열
print(y.shape)  #(10,)   행           x와 y의 행열을 맞춰야함

x = x.transpose()   #동일 x=x.T
print(x.shape)  #(10, 2)

'''
#2. 모델구성 
model = Sequential()
model.add(Dense(50, input_dim=2))  #feature=column
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(350))
model.add(Dense(500))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=200, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss는 " , loss)

result= model.predict([[10,1.6]])
print("10과 1.6의 예측값: ", result)

'''

