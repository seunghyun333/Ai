#feature importance 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test =train_test_split(x,y,train_size=0.8, shuffle=True, random_state=42) # train사이즈는 kfold가 발리데이션이ㄴ;까 포함한 0.8

#kfold train,test 셋 나눌 때 아무데나 넣어도 상관없음. kfold 선언이라서 
n_splits = 5 # 11이면 홀수 단위로 자름 10개가 train, 1개가 test
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) 

# sclaer 적용
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델(conda에서 xgboost 설치)
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) #cv: cross validation

y_predict = cross_val_predict(model, 
                              x_test, y_test,
                              cv=kfold)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)


#catboost: r2 : 0.8148568252525309
#xgb: r2 : 0.7784779387669546
#lgbm: r2 : 0.7959873842146158


#시각화
import matplotlib.pyplot as plt
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()
