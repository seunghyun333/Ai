##from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# reshape
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#[실습] 
#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                 input_shape=(28,28,1), activation='relu'))  #activation='relu'추가 
# model.add(MaxPooling2D(2,2)) 
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3)) #퍼센트로 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=256)

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

###
# maxpooling, dropout 전 
# loss :  0.4495483934879303
# acc :  0.9068999886512756

#Maxpooling2D(2,2), 레이어 2, Dropout(0.2)
# loss :  0.24207457900047302
# acc :  0.9121000170707703

#Maxpooling2D(2,2), 레이어 1, Dropout(0.3)
# loss :  0.2307022511959076
# acc :  0.9283000230789185