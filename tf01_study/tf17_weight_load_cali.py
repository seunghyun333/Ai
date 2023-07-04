#가중치값 저장장

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
#datasets = load_boston()
datasets = fetch_california_housing()
x = datasets.data #변수, 컬럼, 열 
y = datasets.target 

print(datasets.feature_names)
print(datasets.DESCR)

print(x.shape)  #(20640, 8)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=100, shuffle=True
)
print(x_train.shape)    #(14447, 8)
print(y_train.shape)    #(14447,)

#2. 모델구성
model=Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


model.load_weights('./_save/tf17_weight_cali.h5')

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #컴파일은 지우면 안됨됨

'''
## early stopping
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                                verbose=1, restore_best_weights=True )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=200, 
                validation_split=0.2,  
                callbacks=[earlyStopping],
                verbose=1) 
end_time = time.time() - start_time
model.save_weights('./_save/tf17_weight_cali.h5')
'''

#4. 예측,평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)
#print('걸린시간: ', end_time)



#==============================#
#weight save
# Epoch 222: early stopping
# r2 스코어:  0.5424216600004373
# 걸린시간:  26.303983449935913

#weight load
# r2 스코어:  0.5424216600004373