import numpy as np  
from keras.models import Sequential
from keras.layers import Dense


# 1. 데이터 (분석해서 정리,,? )
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])  #앞 부분 학습되어서 y값 6이 나와야함함

# 2. 모델구성
model=Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')  #mae: w값이 음수(-)일때 'mae'사용 
model.fit(x,y,epochs=100)   

# 4.  평가 예측,
loss = model.evaluate(x,y)
print('loss 값은: ', loss)

result = model.predict([6])
print('6의예측값: ', result)