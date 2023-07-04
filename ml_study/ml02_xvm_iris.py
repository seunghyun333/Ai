import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100)

#print(x.shape, y.shape)                 #(150, 4) (150,)
#print('y의 라벨값: ', np.unique(y))     #y의 라벨값:  [0 1 2]

#2. 모델
model = LinearSVC()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
result = model.score(x_test,y_test) 
print('결과 acc:  ', result)

#SVC 값 : 결과 acc:   0.9733333333333334
#SVC train test split  결과 acc:   0.9777777777777777
#LinearSVC() 결과 acc:   0.9666666666666667
#딥러닝 결과 : 찾아보기,,