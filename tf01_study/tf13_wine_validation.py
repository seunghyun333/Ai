#[실습] one hot encoding을 사용하여 분석 (세 가지 방법 중 한 가지 사용)
# validation

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
    x,y,train_size=0.6, test_size=0.2, random_state=100, shuffle=True
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
hist = model.fit(x_train, y_train, epochs=60, batch_size=100, validation_split=0.2)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)

# 예측, 평가
loss, acc = model.evaluate(x_test,y_test)
print('loss: ', loss)
print('acc: ', acc)

#loss:  0.15465861558914185
#acc:  0.9629629850387573

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='orange', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='pink', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()