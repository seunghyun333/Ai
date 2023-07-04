
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


#2. 모델(conda에서 xgboost 설치)
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련 
model.fit(x_train,y_train,
          early_stopping_rounds=20, 
          eval_set=[(x_train,y_train), (x_test,y_test)],
          eval_metric='merror')
            # eval_metric 회귀모델 : rmse, mae, rmsle...
            #             이진분류 : erro r, auc,logloss...
            #             다중분류 : merror, mlogloss...

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

#early: cv predict acc :  0.9629629629629629

#selectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test=selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model=XGBClassifier()
    selection_model.fit(select_x_train,y_train)
    y_predict=selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, ACC:%.2f%%" %(thresh, select_x_train.shape[1], score*100))
    
'''
(124, 8) (54, 8)
Thresh=0.021, n=8, ACC:87.04%
(124, 10) (54, 10)
Thresh=0.014, n=10, ACC:88.89%
(124, 6) (54, 6)
Thresh=0.027, n=6, ACC:87.04%
(124, 11) (54, 11)
Thresh=0.008, n=11, ACC:90.74%
(124, 5) (54, 5)
Thresh=0.043, n=5, ACC:87.04%
(124, 4) (54, 4)
Thresh=0.137, n=4, ACC:87.04%
(124, 1) (54, 1)
Thresh=0.258, n=1, ACC:79.63%
(124, 13) (54, 13)
Thresh=0.000, n=13, ACC:88.89%
(124, 12) (54, 12)
Thresh=0.002, n=12, ACC:88.89%
(124, 2) (54, 2)
Thresh=0.227, n=2, ACC:85.19%
(124, 7) (54, 7)
Thresh=0.023, n=7, ACC:87.04%
(124, 9) (54, 9)
Thresh=0.020, n=9, ACC:88.89%
(124, 3) (54, 3)
Thresh=0.219, n=3, ACC:87.04%
'''