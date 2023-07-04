import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences


(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=10000) #embadding레이어의 input_dim

print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(x_test.shape, y_test.shape)   #(25000,) (25000,)
print(np.unique(y_train, return_counts=True))   #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(len(np.unique(y_train)))  #2   (긍정이다, 부정이다)

# 최대 길이와 평균 길이
print('리뷰의 최대길이 : ', max(len(i) for i in x_train))   #리뷰의 최대길이 :  2494
print('리뷰의 평균길이 : ', sum(map(len, x_train)) / len(x_train)) #리뷰의 평균길이 :  238.71364

print(x_train.shape, y_train.shape) #(25000,) (25000, 2)
print(x_test.shape, y_test.shape)   #(25000,) (25000, 2)


# pad_squences
x_train = pad_sequences(x_train, padding='pre', maxlen=100)  #maxlen이 제일 긴 문장의 단어 수 ? ?
x_test = pad_sequences(x_test, padding='pre', maxlen=100)

print(x_train.shape, y_train.shape)  #(25000, 100) (25000, 2)
print(x_test.shape, y_test.shape)   #(25000, 100) (25000, 2)

#2. 모델구성
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim = 100))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation ='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation ='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train,y_train, epochs=100, batch_size=32,validation_split=0.2)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.693149745464325
# acc :  0.5