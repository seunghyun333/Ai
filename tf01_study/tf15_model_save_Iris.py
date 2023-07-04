#다중분류  아웃풋 3개,activation='softmax'
#validation #early stopping

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import time

#1. 데이터 
datasets = load_iris()
print(datasets.DESCR)   # 상세정보 
print(datasets.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


x = datasets['data']    # 동일 문법  x = datasets.data
y = datasets.target 
print(x.shape, y.shape)     #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) #(105, 4) (105,)
print(x_test.shape, y_test.shape)   #(45, 4) (45,)
print(y_test)   #0은 꽃1, 1은 꽃2, 2는 꽃3

#2. 모델링
model=Sequential()
model.add(Dense(100, input_dim=4)) #컬럼 수
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))     #아웃풋 3개

model.save('./_save/tf15_iris.h5') # h5로 모델 저장 ########################

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 
#loss='sparse_categorical_crossentropy' 원핫인코딩 안하고 0.1.2 정수로 인코딩 상태에서 수행가능

##early stopping
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                                verbose=1, restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2,
                callbacks=[earlystopping], verbose=1)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)



#4. 예측, 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린 시간: ', end_time)


# accuracy: 1.0000 => 과적합일 가능성이 높음 
# loss :  0.020110951736569405

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='black', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

#===================#
# save 값 
#epochs 1000, patience 10
# Epoch 450: early stopping
# loss :  0.016567150130867958
# acc :  1.0
# 걸린 시간:  11.985859394073486