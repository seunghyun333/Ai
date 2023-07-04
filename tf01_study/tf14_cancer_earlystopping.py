#이진분류= 회귀 = 캘리포니아
#early stopping
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score  # r2=적합도, ac=정확도 
from sklearn.datasets import load_breast_cancer
import time

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
x = datasets.data   
y = datasets.target     
print(x.shape, y.shape)     #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

#2. 모델구성
model=Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #0이냐 1이냐 찾는 이진분류이기때문에 뉴런 1, 이진분류는 무조건 아웃풋 활성화 함수를: sigmoid

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mse']) #손실함수는 바이너리 넣어야함, metrics, accuracy 역할할 확인 하기**

## early stopping
from keras.callbacks import EarlyStopping
earlystoppinng = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                                verbose=1, restore_best_weights=True)
start_time= time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=200,
        validation_split=0.2,
        callbacks=[earlystoppinng],
        verbose=1)
end_time = time.time() - start_time

#4. 평가 예측  
# loss =model.evaluate(x_test,y_test)
loss, acc, mse = model.evaluate(x_test, y_test)
y_predict =model.predict(x_test) #소수점 나와서 0인지 1인지 만들어줘야함

y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict) 
print('loss : ', loss)  #loss :  [0.3225121796131134, 0.9064327478408813, 0.08121472597122192]
print('accuracy : ', acc)   #accuracy :  0.9064327485380117
print('걸린 시간: ', end_time) #걸린 시간:  1.084829568862915

#시그모이드랑 바이너리 넣으면 끝, 분류기 때문에 r2scoore가 아니라 accurancy score 체크

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='orange', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

#==============================#
#epochs 5000, patience 50 일 때
# Epoch 102: early stopping
# loss :  0.26109713315963745
# accuracy :  0.9473684210526315
# 걸린 시간:  2.953126907348633

#epochs 5000, patience 100 일 때
# Epoch 265: early stopping
# loss :  0.19277802109718323
# accuracy :  0.9385964912280702
# 걸린 시간:  7.374332427978516

#epochs 5000, patience 200 일 때
# Epoch 718: early stopping
# loss :  0.1545599400997162
# accuracy :  0.9385964912280702
# 걸린 시간:  19.094697952270508
