import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=42, shuffle=True)


# 스케일러
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#KFold
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#parameter
param ={
    'n_estimators': [100],
    'random_state':[42,62,72],
    'max_features':[3,4,7]
}

#2. 모델(Bagging)
bagging = BaggingRegressor(DecisionTreeRegressor(), 
                            n_estimators=100,
                            n_jobs=-1,
                            random_state=42)
model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time() - start_time

#4. 평가,예측
result = model.score(x_test,y_test)

print('최적의 매개변수 : ', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
print('걸린시간: ', end_time, '초')
print('bagging 결과 : ', result)

#배깅만 했을 때
# 걸린시간:  3.731471300125122 초
# bagging 결과 :  0.8042348744721923

#kfold, grid 적용결과
# 최적의 매개변수 :  BaggingRegressor(estimator=DecisionTreeRegressor(), max_features=7,
#                  n_estimators=100, n_jobs=-1, random_state=42)
# 최적의 파라미터 :  {'max_features': 7, 'n_estimators': 100, 'random_state': 42}
# 걸린시간:  49.88687562942505 초
# bagging 결과 :  0.822087951242456


