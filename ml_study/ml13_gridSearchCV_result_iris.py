import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)

# Scaler 적용
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier(max_depth=6, n_estimators= 200, n_jobs= -1)
#refit 기본값은 False라 꼭 True로 해줘야함: 위에 입력한 파라미터를 찾아서 적용하는 기능

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time= time.time() - start_time

print('걸린 시간 : ', end_time, '초')

# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 200, 'n_jobs': -1}
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_estimators=200, n_jobs=-1)
# best_score :  0.95
# model_score :  1.0

# 그리드 서치 후
# 걸린 시간 :  0.2216627597808838 초
# cv pred acc :  0.9666666666666667


#4. 평가, 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) # cv : cross validation
# print('cv acc : ', score)
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
# print('cv pred : ', y_predict)
acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

# 결과 
# SVC() 결과 acc : 0.9777777777777777
# MinMaxScaler : 0.9777777777777777
# ==================================
# tree 결과 acc :  0.9555555555555556
# ==================================
# ensemble 결과 acc :  0.9555555555555556
# kfold 결과 acc :  1.0
