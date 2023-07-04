
# 3.wine    다중분류 SVC, LinveraSVC
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수 값 조정 

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=100)

#kfold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
# sclaer 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델(conda에서 lightgbm 설치)
from lightgbm  import LGBMClassifier
model = LGBMClassifier()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
score = cross_val_score(model,
                        x_train, y_train,
                        cv=kfold)
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
acc = accuracy_score(y_test, y_predict)

print("cross validation acc: ",score)
#[0.92       0.92       0.96       0.96       0.95833333]
print('cv predict : ', y_predict)
#y_test 만큼 개수 나옴
print('cv predict acc : ', acc) 
#cv predict acc :  0.9814814814814815

#SVC 결과 acc:  0.5555555555555556
#Linear SVC 결과 acc:   0.6851851851851852
#딥러닝: accuracy :  0.7777777910232544

#Scaler 적용 
#Standard : 0.9814814814814815
#Minmax : 0.9814814814814815
#MaxAbs: 0.9814814814814815
#Robuster:   0.9814814814814815

#decision tree
#Minmax :0.9814814814814815
#MaxAbs: 0.9814814814814815

#앙상블
#MaxAbs 결과 acc:   1.0
#Minmax 결과 acc:   1.0

#xgb :  0.9629629629629629
#lgbm:  0.9629629629629629
#catboost : 0.9629629629629629