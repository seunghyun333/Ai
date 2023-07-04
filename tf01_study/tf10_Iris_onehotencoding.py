#다중분류  아웃풋 3개,activation='softmax', 원핫인코딩

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

##모델구성전에 원핫인코딩하기- 3가지 방법잇음 
###one hot encoding
from keras.utils import to_categorical
y = to_categorical(y) # 끝
print(y)        #[1. 0. 0.] '''
print(y.shape) #(150, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) #(105, 4) (105,)
print(x_test.shape, y_test.shape)   #(45, 4) (45,)
print(y_test)   #0은 꽃1, 1은 꽃2, 2는 꽃3





#2. 모델링
model=Sequential()
model.add(Dense(100, input_dim=4)) #컬럼 수
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))     #아웃풋 3개

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린 시간: ', end_time)

#4. 예측, 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)


# loss :  0.05172388255596161
# acc :  0.9777777791023254
