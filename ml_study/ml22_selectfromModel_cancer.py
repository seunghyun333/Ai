# 2.cancer 이진분류 SVC, LinveraSVC
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=42, shuffle=True)

#kfold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# sclaer 적용
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델(conda에서 xgboost 설치)
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련 
model.fit(x_train,y_train,
          early_stopping_rounds=20, 
          eval_set=[(x_train,y_train), (x_test,y_test)],
          eval_metric='error')
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
acc=accuracy_score(y_test, y_predict)

print("cross validation acc: ",score)
#[0.95       0.9625     0.95       0.97468354 0.97468354]
print('cv predict : ', y_predict)
#y_test 개수 만큼 나옴 
print('cv predict acc : ', acc) 


#early stopping
#cv predict acc :  0.9707602339181286

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
(398, 20) (171, 20)
Thresh=0.006, n=20, ACC:97.66%
(398, 14) (171, 14)
Thresh=0.012, n=14, ACC:97.66%
(398, 6) (171, 6)
Thresh=0.038, n=6, ACC:96.49%
(398, 9) (171, 9)
Thresh=0.023, n=9, ACC:96.49%
(398, 21) (171, 21)
Thresh=0.006, n=21, ACC:98.25%
(398, 30) (171, 30)
Thresh=0.000, n=30, ACC:98.25%
(398, 30) (171, 30)
Thresh=0.000, n=30, ACC:98.25%
(398, 1) (171, 1)
Thresh=0.329, n=1, ACC:89.47%
(398, 30) (171, 30)
Thresh=0.000, n=30, ACC:98.25%
(398, 26) (171, 26)
Thresh=0.002, n=26, ACC:97.66%
(398, 10) (171, 10)
Thresh=0.017, n=10, ACC:97.66%
(398, 24) (171, 24)
Thresh=0.005, n=24, ACC:98.25%
(398, 11) (171, 11)
Thresh=0.017, n=11, ACC:97.66%
(398, 16) (171, 16)
Thresh=0.010, n=16, ACC:97.66%
(398, 27) (171, 27)
Thresh=0.002, n=27, ACC:98.25%
(398, 18) (171, 18)
Thresh=0.008, n=18, ACC:97.66%
(398, 7) (171, 7)
Thresh=0.031, n=7, ACC:95.91%
(398, 25) (171, 25)
Thresh=0.002, n=25, ACC:97.66%
(398, 22) (171, 22)
Thresh=0.005, n=22, ACC:97.66%
(398, 12) (171, 12)
Thresh=0.014, n=12, ACC:97.08%
(398, 5) (171, 5)
Thresh=0.048, n=5, ACC:94.74%
(398, 8) (171, 8)
Thresh=0.031, n=8, ACC:97.08%
(398, 3) (171, 3)
Thresh=0.057, n=3, ACC:93.57%
(398, 4) (171, 4)
Thresh=0.055, n=4, ACC:95.32%
(398, 23) (171, 23)
Thresh=0.005, n=23, ACC:98.25%
(398, 19) (171, 19)
Thresh=0.007, n=19, ACC:97.08%
(398, 13) (171, 13)
Thresh=0.014, n=13, ACC:95.91%
(398, 2) (171, 2)
Thresh=0.235, n=2, ACC:88.30%
(398, 17) (171, 17)
Thresh=0.009, n=17, ACC:97.66%
(398, 15) (171, 15)
Thresh=0.011, n=15, ACC:97.66%
'''