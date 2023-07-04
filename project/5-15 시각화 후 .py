#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


# In[73]:


# 1. 데이터
path = './_data2/'
datasets = pd.read_csv (path + 'train.csv')

# print (datasets.columns)
# print ('\n-----------------\n')
# print (datasets.head(7))

  # 입력 변수들의 데이터프레임
filtered_data = datasets[datasets['date'].isin(['2019-09-01', '2019-09-07','2019-09-07','2019-09-08','2019-09-12','2019-09-13',
                                                '2019-09-14','2019-09-15','2019-09-21','2019-09-22','2019-09-28','2019-09-29']) == False]


# 필터링된 데이터를 x, y로 분할
x = filtered_data[['date', 'bus_route_id', 'in_out', 'station_code', 'station_name',
       'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
       '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
       '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']]
y = filtered_data[['18~20_ride']] 

x = x.drop(columns=[x.columns[2]])


print (x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=42, shuffle=True)

print (x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#print(pd.isna(x))  #NaN값이 있는 경우 True, 없는 경우 False
print (x.info())


# In[74]:


# LabelEncoder를 적용할 열
cols_to_encode = ['date', 'station_name']

# LabelEncoder 객체 생성 및 변환
label_encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    x_train.loc[:, col] = le.fit_transform(x_train[col])
    label_encoders[col] = le
    

for col in cols_to_encode:
    le = LabelEncoder()
    x_test.loc[:, col] = le.fit_transform(x_test[col])
    label_encoders[col] = le


# In[29]:


# print("\nAfter Label Encoding:")
# print(x_test)


# In[30]:


# print(x.isnull().sum())  #누락된 데이터 개수
# print (x.describe()) #카운트 전체데이터수 , mean 평균, std 표준오차


# In[75]:


x_train = x_train.fillna(0)
x_test = x_test.fillna(0)
# print (x_test)


# In[32]:


# ### 상관 계수 히트맵 (시각화) ###
# import matplotlib.pyplot as plt
# import seaborn as sns


# sns.set (font_scale = 14.0)
# sns.set (rc = {'figure.figsize' : (27, 18)})   # 중괄호 dictionary
# sns.heatmap (data = datasets.corr(), square = True, \
#              annot = True, cbar = True) # 상관관계
# plt.show ()


# In[76]:


#Scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = CatBoostRegressor()


# In[77]:


# #earlystopping
# from keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True)

y_train = np.ravel(y_train)
model.fit(x_train,y_train)


# In[78]:


from sklearn.model_selection import KFold
#kfold train,test 셋 나눌 때 아무데나 넣어도 상관없음. kfold 선언이라서 
n_splits = 5 # 11이면 홀수 단위로 자름 10개가 train, 1개가 test
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) 


# In[79]:


from sklearn.model_selection import cross_val_score, cross_val_predict

score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold) #cv: cross validation

y_predict = cross_val_predict(model, 
                              x_test, y_test,
                              cv=kfold)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2) 
# robust, cat r2 : 0.6030266805228426
# minmax, cat r2 : 0.6032297720325435
# StandardScaler, cat  r2 : 0.6031417044605571
# minmax, lgbm r2 : 0.5742431839084499
# importance 후 in_out 삭제, minmax, cat r2 : 0.6077579210066286
# importance 후 in_out &6-7 takeoff 삭제, minmax, cat r2 : 0.6112568007875296


# In[12]:


#시각화

import matplotlib.pyplot as plt

n_features = x.shape[1]
plt.barh(range(n_features), model.feature_importances_)
plt.yticks(np.arange(n_features), x.columns)
plt.title("Feature Importance - Bus Ride")
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.ylim(-1, n_features)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
#x_test에서 ride값 출력해봐야함

from sklearn.preprocessing import LabelEncoder, StandardScaler

# test 데이터
path = './_data2/'
test_datasets = pd.read_csv(path + 'test.csv')

# 필터링된 데이터를 x로 분할
x1 = test_datasets[['id', 'date', 'bus_route_id', 'in_out', 'station_code', 'station_name',
       'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
       '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
       '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']]

# LabelEncoder를 적용할 열을 선택합니다.
cols_to_encode = ['date', 'in_out', 'station_name']

# LabelEncoder 객체를 생성하고 열을 순회하며 변환합니다.
label_encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    x1.loc[:, col] = le.fit_transform(x1.loc[:, col])
    label_encoders[col] = le

# 모델의 predict() 메서드 호출을 위해 스케일링을 수행합니다.
scaler = StandardScaler()
numerical_cols = ['latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
                  '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
                  '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']
x1.loc[:, numerical_cols] = scaler.fit_transform(x1.loc[:, numerical_cols])

# 데이터의 차원 확인
x1 = x1.drop (['longitude', 'latitude'], axis = 1)
print(x1.shape)

# 모델의 predict() 메서드 호출
y_predict = model.predict(x1)

# 예측 결과를 test 데이터프레임의 '18~20_ride' 컬럼에 저장합니다.
test_datasets['18~20_ride'] = y_predict

# 예측 결과 출력
print(test_datasets['18~20_ride'])
'''


# In[ ]:




