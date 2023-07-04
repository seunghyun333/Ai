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
acc=accuracy_score(y_test, y_predict)

print("cross validation acc: ",score)
#[0.95       0.9625     0.95       0.97468354 0.97468354]
print('cv predict : ', y_predict)
#y_test 개수 만큼 나옴 
print('cv predict acc : ', acc) 
#cv predict acc :  0.9532163742690059

#SVC 결과 acc:   0.9064327485380117
#Linear SVC 결과 acc:   0.9415204678362573
#딥러닝: accuracy :  0.9064327485380117


#Scaler 적용 
#Standard : 0.9473684210526315
#Minmax : 0.9649122807017544
#MaxAbs: 0.9649122807017544
#Robuster:    0.9532163742690059

#decision tree
#minmax: 결과 acc:   0.935672514619883
#MaxAbs: 0.9415204678362573
#Robuster:    0.9298245614035088

#앙상블
#Robuster: 결과 acc:   0.9649122807017544

#catboost: 0.9590643274853801
# lgbm:0.9415204678362573
# xgb: 
