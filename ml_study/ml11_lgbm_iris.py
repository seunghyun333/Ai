import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_iris()
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
from lightgbm  import LGBMClassifier
model = LGBMClassifier()


#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) #cv: cross validation
print("cross validation acc: ",score)
#cv acc 는 kfold의 n splits 만큼 값이 나옴 . 5여서 [0.91666667 1.         0.91666667 0.83333333 1.        ]

y_predict = cross_val_predict(model, 
                              x_test, y_test,
                              cv=kfold)
print('cv predict : ', y_predict)
#0.2 y_test 만큼 개수 나옴: cv predict :  [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 1 0 2 2 2 2 2 0 0]

acc = accuracy_score(y_test, y_predict)
print('cv predict acc : ', acc)
# cv predict acc :  0.9666666666666667



#SVC 값 : 결과 acc:   0.9733333333333334
#SVC train test split  결과 acc:   0.9777777777777777
#LinearSVC() 결과 acc:   0.9666666666666667
#딥러닝 결과 : 찾아보기,,

#Scaler 적용 
#Standard : 0.9333333333333333
#Minmax : 0.9333333333333333
#MaxAbs: 0.9555555555555556
#Robuster:    0.9333333333333333

#decision tree
#scaler: minmax 결과 acc:   0.9555555555555556

#앙상블  scaler: minmax 결과 acc: 0.9555555555555556

#kfold: n 11,3,5 결과 acc:   1.0

#train,test set kfold  - cv predict acc :  0.9666666666666667

#xgboost : 0.8666666666666666666666
#lgbm :  0.2
