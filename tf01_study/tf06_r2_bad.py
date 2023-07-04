#[실습]
#1. r2score를 음수가 아니라 0.5 이하로 만들어 보세요
#2. 데이터는 건드리지 마세요
#3. 레이어는 인풋, 아웃풋 포함7개 이상으로 만들어주세요
#4. batch_size=1
#5. 히든레이어의 노드 갯수는 10개 이상 100개 이하
#6. train_size = 0.7
#7. epochs=100 이상
#[실습시작]

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
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss는', loss)

y_predict = model.predict(x)

### R2score
from sklearn.metrics import r2_score, accuracy_score #회귀는 r2, 분류는 acc
r2 = r2_score(y, y_predict)  #실제 y값이라 예측 y값 비교
print('r2스코어 : ', r2)

#result
#노드수를 늘리고 히든레이어의 갯수를 늘리면 r2 score가 좋지 않다





