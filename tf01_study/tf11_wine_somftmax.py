# 실습 loss = 'sparse_categorical_crossentropy'를 사용하여 분석 

from tracemalloc import start
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import time

#1. 데이터
datasets = load_wine()
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(178, 13) (178,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) #(124, 13) (124,)
print(x_test.shape, y_test.shape)   #(54, 13) (54,)
print(y_test)   #[0 1 2]

#2. 모델링
model=Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린시간: ', end_time)

# 예측, 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc : ', acc)

#걸린시간:  2.6191811561584473
# loss : 0.2452908307313919
# acc :  0.9444444179534912