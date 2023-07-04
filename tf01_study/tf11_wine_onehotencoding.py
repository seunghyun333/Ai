#[실습] one hot encoding을 사용하여 분석 (세 가지 방법 중 한 가지 사용)

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
x = datasets.data
y = datasets.target

from keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.7, random_state=100, shuffle=True
)

#2. 모델링
model=Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)

# 예측, 평가
loss, acc = model.evaluate(x_test,y_test)
print('loss: ', loss)
print('acc: ', acc)

#loss:  0.15465861558914185
#acc:  0.9629629850387573