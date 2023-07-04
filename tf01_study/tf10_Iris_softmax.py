#다중분류  아웃풋 3개,activation='softmax'

from tracemalloc import start
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
    x, y, train_size=0.7, random_state=100, shuffle=True
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

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 
#loss='sparse_categorical_crossentropy' 원핫인코딩 안하고 0.1.2 정수로 인코딩 상태에서 수행가능
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)

#4. 예측, 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)


# accuracy: 1.0000 => 과적합일 가능성이 높음 
# loss :  0.020110951736569405
