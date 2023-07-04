# 2.cancer 이진분류 SVC, LinveraSVC

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100)

#print(x.shape, y.shape) #(569, 30) (569,)

# sclaer 적용
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델
model = DecisionTreeClassifier()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
result = model.score(x_test,y_test) 
print('결과 acc:  ', result)

#SVC 결과 acc:   0.9064327485380117
#Linear SVC 결과 acc:   0.9415204678362573
#딥러닝: accuracy :  0.9064327485380117


#Scaler 적용 
#Standard : 0.9473684210526315
#Minmax : 0.9649122807017544
#MaxAbs: 0.9649122807017544
#Robuster:    0.9532163742690059

#decision tree
#minmax: 결과 acc:   0.935672514619883
#MaxAbs: 0.9415204678362573
#Robuster:    0.9298245614035088
