#feature importance 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedGroupKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test =train_test_split(x,y,train_size=0.8, shuffle=True, random_state=42) # train사이즈는 kfold가 발리데이션이ㄴ;까 포함한 0.8

#StratifiedGroupKFold   train,test 셋 나눌 때 아무데나 넣어도 상관없음. kfold 선언이라서 
n_splits = 5 # 11이면 홀수 단위로 자름 10개가 train, 1개가 test
random_state = 42
kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state) 

# sclaer 적용
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = DecisionTreeClassifier()

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가 예측




'''

###feature importances 
print(model, " : ", model.feature_importances_)

#시각화
import matplotlib.pyplot as plt 
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_)
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title("wine feature importances")
plt.ylabel('Feature')
plt.xlabel('importance')
plt.ylim(-1, n_features) #가로로 보기

plt.show()
'''