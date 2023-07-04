#모델체크 포인트 로드. 모델을 다 지워도 됨 ,, ,!!! 모델과 같음 

import numpy as np
from keras.models import Sequential, load_model ##################추가가
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
#datasets = load_boston()
datasets = fetch_california_housing()
x = datasets.data  
y = datasets.target 

print(datasets.feature_names)
print(datasets.DESCR)

print(x.shape) 
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=100, shuffle=True
)
print(x_train.shape)   
print(y_train.shape)   


# #2. 모델구성
# model=Sequential()
# model.add(Dense(100, input_dim=8))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(1))



# #3. 컴파일,훈련
# model.compile(loss='mse', optimizer='adam')

## early stopping
from keras.callbacks import EarlyStopping, ModelCheckpoint      ### 모델체크포인트 추가가
# earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                                # verbose=1, restore_best_weights=True )
        
# Model Check Point - 따로 보관하기기
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_weights_only=True,
#     filepath='./_mcp/tf18_cali.hdf5'  #파일명 체크 
# )

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=5000, batch_size=200, 
#                 validation_split=0.2,  
#                 callbacks=[earlyStopping, mcp],
#                 verbose=1) 
# end_time = time.time() - start_time

model = load_model('./_mcp/tf18_cali.hdf5')

#4. 예측,평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)
# print('걸린시간: ', end_time)



#==============================#
# Epoch 321: early stopping
# r2 스코어:  0.5486243827633063
# 걸린시간:  43.54737997055054

#load
# r2 스코어:  0.541111940525899