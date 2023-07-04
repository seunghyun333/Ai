import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import time

#1. 데이터
(x_train, y_train), (x_test, y_test)= cifar10.load_data() # x이미지 y라벨
# print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)#(10000, 32, 32, 3) (10000, 1)


#정규화 Nomalization
x_train = x_train/255.0
x_test = x_test/255.0

#2. 모델 구성
model= Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),
                padding='same',
                activation='relu',
                input_shape=(32,32,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
from keras.callbacks import EarlyStopping
earlystoppinng = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                                verbose=1, restore_best_weights=True)
start_time= time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=200,
        validation_split=0.2,
        callbacks=[earlystoppinng],
        verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('acc :', acc)
print('걸린 시간 : ', end_time)

######
