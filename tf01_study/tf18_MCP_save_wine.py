#[실습] one hot encoding을 사용하여 분석
# validation & earlystopping

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

## early stopping

from keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                                verbose=1, restore_best_weights=True)

                              
#Model Check point
mcp = ModelCheckpoint(
    monitor="val_loss",
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_mcp/tf18_wine.hdf5'
)                    
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2,
                callbacks=[earlystopping, mcp], verbose=1)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)


# 예측, 평가
loss, acc = model.evaluate(x_test,y_test)
print('loss: ', loss)
print('acc: ', acc)



#==============================#
#save
# Epoch 265: early stopping
# 걸린 시간:  8.108789443969727
# loss:  0.4579636752605438
# acc:  0.7777777910232544