#validation_split & ealry stopping & model_save
#import 추가: from keras.models import Sequential, load_model

import numpy as np
from keras.models import Sequential, load_model
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
# model=Sequential()
# model.add(Dense(100, input_dim=8))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(1))
# model.save('./_save/tf15_cali.h5') # h5로 모델 저장
model = load_model('./_save/tf15_cali.h5')


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

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

#4. 예측,평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)
print('걸린시간: ', end_time)



#==============================#
#epochs=5000, patience 50  일 때

# save model 값 
# Epoch 149: early stopping
# r2 스코어:  0.5297356286390125
# 걸린시간:  22.609537601470947

# load model 값
# Epoch 83: early stopping
# r2 스코어:  0.4987951497314347
# 걸린시간:  10.350826263427734