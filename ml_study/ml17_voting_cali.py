import numpy as np 
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=True, random_state=42)

# 스케일러
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델(voting)-모델에 그리드서치한 값을 각 클래스에 추가해서 돌리기 
xgb = XGBRegressor()
lgbm = LGBMRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(estimators=[('xgb', xgb),('lgbm', lgbm),('cat', cat)], # 모델이랑 이름 같이 넣어줘야함
                         n_jobs=-1) #regressor에선 괄호안에 voting 없음 , 하드 소프트는: 분류 모델에서만 사용. 

#3. 훈련
model.fit(x_train,y_train)

# 4. 평가 예측
# y_predict = model.predict(x_test)
# score = accuracy_score(y_test, y_predict)
# print('voting 결과: ', score)

regressors = [cat,xgb,lgbm]
for model in regressors:
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0} 정확도: {1: .4f}'.format(class_name, score))


# CatBoostRegressor 정확도:  0.8492
# XGBRegressor 정확도:  0.8287
# LGBMRegressor 정확도:  0.8365