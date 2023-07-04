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
model = XGBRegressor(random_state=123, n_estimators=1000,
                    learning_rate = 0.1, max_depth = 6, gamma= 1)

#3. 훈련 
model.fit(x_train,y_train,
          early_stopping_rounds=20, 
          eval_set=[(x_train,y_train), (x_test,y_test)],
          eval_metric='rmse')
            # eval_metric 회귀모델 : rmse, mae, rmsle...
            #             이진분류 : erro r, auc,logloss...
            #             다중분류 : merror, mlogloss...

#4. 평가 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) #cv: cross validation

y_predict = cross_val_predict(model, 
                              x_test, y_test,
                              cv=kfold)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

# cv pred r2 : 0.7767990598223357

#selectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test=selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model=XGBRegressor()
    selection_model.fit(select_x_train,y_train)
    y_predict=selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2:%.2f%%" %(thresh, select_x_train.shape[1], score*100))

#r2 : 0.7767990598223357

# selectfrom model
# (16512, 1) (4128, 1)
# Thresh=0.545, n=1, R2:44.92%
# (16512, 5) (4128, 5)
# Thresh=0.071, n=5, R2:82.91%
# (16512, 6) (4128, 6)
# Thresh=0.037, n=6, R2:83.49%
# (16512, 7) (4128, 7)
# Thresh=0.024, n=7, R2:83.35%
# (16512, 8) (4128, 8)
# Thresh=0.022, n=8, R2:82.87%
# (16512, 2) (4128, 2)
# Thresh=0.148, n=2, R2:54.73%
# (16512, 4) (4128, 4)
# Thresh=0.073, n=4, R2:82.02%
# (16512, 3) (4128, 3)