#validation_split & ealry stopping

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #변수, 컬럼, 열 
y = datasets.target 

print(datasets.feature_names)

print(x.shape)  #(20640, 8)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)
print(x_train.shape)    #(14447, 8)
print(y_train.shape)    #(14447,)

#2. 모델구성
model=Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1)) 

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

## early stopping
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', #val 기준으로 멈춰! 처음에는 100번정도,, 작게하면 다음에 튈지도 모름, mode기본값은 auto.
                                verbose=1, restore_best_weights=True ) #restore~ 기본값: False이므로 가장 좋은 값 나오면 멈출거니? 응 True 체크크
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=200, 
            validation_split=0.2,  
            callbacks=[earlyStopping],
            verbose=1)  #1 보겠다 
end_time = time.time() - start_time

#4. 예측,평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)
print('걸린시간: ', end_time)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.',c='orange', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

#==============================#
#epochs 500 일때 
# r2 스코어:  0.5423435655504028
# 걸린시간:  35.26150369644165

#epochs 5000, patience 100 일 때
# Epoch 579: early stopping
# 129/129 [==============================] - 0s 943us/step - loss: 0.6005
# 129/129 [==============================] - 0s 804us/step
# r2 스코어:  0.5543514821537454
# 걸린시간:  57.997618675231934

#epochs 5000, patience 50  일 때
# Epoch 354: early stopping
# 129/129 [==============================] - 0s 901us/step - loss: 0.6071
# 129/129 [==============================] - 0s 784us/step
# r2 스코어:  0.5494379245070419
# 걸린시간:  34.4178900718689