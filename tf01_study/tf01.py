# 딥러닝 순서
# 1. 데이터
import numpy as np
x = np.array([1,2,3])  #어레이는 리스트로 작성해야함, x와 y의 행열이 맞아야함함
y = np.array([1,2,3])   

# 2. 모델구성
from keras.models import Sequential  # 파이썬 버전에 따라 가져오는 형식이 다름 
from keras.layers import Dense      #

model = Sequential()
model.add(Dense(10, input_dim=1))    #입력층 -- 뭔지 개념 정리 
model.add(Dense(50))                 #히든레이어- 많다고 좋은건 아님 데이터가 적으면 히든레이어도 적은게 나음음
model.add(Dense(80))                 #뉴런이 많을수록 계산을 잘하는 건 맞지만 ,, 
model.add(Dense(90))                #간단 데이터를 할때 뉴런많이하면 쓸때없는값까지 계산해서 과적합 일어남
model.add(Dense(100))
model.add(Dense(1))                  #출력층 

# 3. 컴파일 & 훈련
model.compile(loss='mse', optimizer='adam')  #로스랑 옵티 종류 찾아보기 
model.fit(x, y, epochs=500)   #에폭은 훈련양 #갔다가 다시 돌아오는게 에폭 1번 , 할때마다 값이 다르기때문에 여러번 하는거임 , 돌아갈때 미분으로 계산함 ; 

# 4. 예측 & 평가
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])   ## 4값 예측
print('4의 예측값', result)   # 윗줄의 4는 x, result는 y

# loss :  0.0
# 4의 예측값 [[4.]]
