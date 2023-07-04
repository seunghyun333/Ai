import numpy as np 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=True, random_state=42)

# 스케일러
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델(voting)
xgb = XGBClassifier()
lgbm = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(estimators=[('xgb', xgb),('lgbm', lgbm),('cat', cat)],# 모델이랑 이름 같이 넣어줘야함
                         voting = 'soft',
                         n_jobs=-1) 

#3. 훈련
model.fit(x_train,y_train)

# 4. 평가 예측
# y_predict = model.predict(x_test)
# score = accuracy_score(y_test, y_predict)
# print('voting 결과: ', score)

classifiers = [cat,xgb,lgbm,]
for model in classifiers:
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0} 정확도: {1: .4f}'.format(class_name, score))

#hard voting ,soft 동일 
# CatBoostClassifier 정확도:  1.0000
# XGBClassifier 정확도:  1.0000
# LGBMClassifier 정확도:  1.0000