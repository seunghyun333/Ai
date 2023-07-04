#feature importance 
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

#StratifiedGroupKFold
n_splits = 5 # 11이면 홀수 단위로 자름 10개가 train, 1개가 test
random_state = 42
kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state) 

# sclaer 적용
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#2. 모델
model = RandomForestClassifier()

#3. 훈련 
model.fit(x,y)

#4. 평가 예측
result = model.score(x,y) 
print('결과 acc:  ', result)





###feature importances 
print(model, " : ", model.feature_importances_)

#시각화
import matplotlib.pyplot as plt 
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_)
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title("Iris feature importances")
plt.ylabel('Feature')
plt.xlabel('importance')
plt.ylim(-1, n_features) #가로로 보기

plt.show()
