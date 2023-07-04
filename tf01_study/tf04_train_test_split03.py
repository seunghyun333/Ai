#라이브러리 구성, 정리
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3, #보통 30퍼
    train_size=0.7, #보통 70퍼 
    random_state=100, # 데이터를 난수값에 의해 추출한다는 의미, 중요한 하이퍼파라미터임임
    shuffle=True    #데이터를 섞어서 가지고 올 것인지 정함
)
#테스트를 넣어도 되고 트레인 넣어도 되고 둘다 넣으려면 100을 맞춰야함
#random_state: 난수값을 지정해줌. 랜덤을 정해줌. 그래서 돌릴때마다 값이 변하지 않음
#shuffle 리스트를 섞을지 말지 /불리언(안적으면 기본값 true)(안하고 싶음 False)

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

result = model.predict([21])
print('21의 예측값 : ', result)


