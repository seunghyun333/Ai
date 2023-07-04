#[실습] one hot encoding을 사용하여 분석
# validation & earlystopping

import numpy as np
from keras.models import Sequential, load_model
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
# model=Sequential()
# model.add(Dense(100, input_dim=13))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

# model.save('./_save/tf15_wine.h5') # h5로 모델 저장 ########################

model = load_model('./_save/tf15_wine.h5')

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

## early stopping

from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                                verbose=1, restore_best_weights=True)

                              
                            
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2,
                callbacks=[earlystopping], verbose=1)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)


# 예측, 평가
loss, acc = model.evaluate(x_test,y_test)
print('loss: ', loss)
print('acc: ', acc)

#loss:  0.15465861558914185
#acc:  0.9629629850387573


#==============================#
#epochs 1000, patience 100일 때

#save
# Epoch 712: early stopping
# 걸린 시간:  21.531862020492554
# loss:  0.6129447817802429
# acc:  0.9444444179534912

#load
# Epoch 699: early stopping
# 걸린 시간:  18.851384162902832
# loss:  1.084640622138977
# acc:  0.8611111044883728
