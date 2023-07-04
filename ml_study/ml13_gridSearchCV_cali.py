import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)

# Scaler 적용
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

param = [
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

#2. 모델
from sklearn.model_selection import GridSearchCV
rf_model = RandomForestRegressor()
model = GridSearchCV(rf_model, param, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1) #refit 기본값은 False라 꼭 True로 해줘야함: 위에 입력한 파라미터를 찾아서 적용하는 기능

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time= time.time() - start_time

print('최적의 파라미터 : ' , model.best_params_)
print('최적의 매개변수 : ' , model.best_estimator_)
print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')

#4. 평가, 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) # cv : cross validation
# print('cv acc : ', score)
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
# print('cv pred : ', y_predict)
acc = r2_score(y_test, y_predict)
print('r2 : ', acc)