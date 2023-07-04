import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets= fetch_california_housing()
x= datasets.data
y= datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7, random_state=123, shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
allAlogirhms = all_estimators(type_filter='regressor')
print(len(allAlogirhms)) #55

#3. 출력(평가, 예측)
for(name, algorithm) in allAlogirhms:
    try: 
        model= algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, "의 정답률 : ", r2)
    except:
        print(name, "안 나온 놈 !!!")
        
        
#######결과 
# ARDRegression 의 정답률 :  0.6093418418782462
# AdaBoostRegressor 의 정답률 :  0.44981282969225267
# BaggingRegressor 의 정답률 :  0.7869258792960991
# BayesianRidge 의 정답률 :  0.6093292143350891
# CCA 안 나온 놈 !!!
# DecisionTreeRegressor 의 정답률 :  0.6380520694813592
# DummyRegressor 의 정답률 :  -0.0003759104259155599
# ElasticNet 의 정답률 :  -0.0003759104259155599
# ElasticNetCV 의 정답률 :  0.6002377292053442
# ExtraTreeRegressor 의 정답률 :  0.5374446240742778
# ExtraTreesRegressor 의 정답률 :  0.8218074284582739
# GammaRegressor 의 정답률 :  0.01922288967930119
# GaussianProcessRegressor 의 정답률 :  -518.4540417454557
# GradientBoostingRegressor 의 정답률 :  0.7967641188178963
# HistGradientBoostingRegressor 의 정답률 :  0.8411685421639945
# HuberRegressor 의 정답률 :  0.5971425101463328
# IsotonicRegression 안 나온 놈 !!!
# KNeighborsRegressor 의 정답률 :  0.690612293462256
# KernelRidge 의 정답률 :  0.5310061849718803
# Lars 의 정답률 :  0.6093875757405811
# LarsCV 의 정답률 :  0.6057018072586755
# Lasso 의 정답률 :  -0.0003759104259155599
# LassoCV 의 정답률 :  0.6062641778423064
# LassoLars 의 정답률 :  -0.0003759104259155599
# LassoLarsCV 의 정답률 :  0.6057018072586755
# LassoLarsIC 의 정답률 :  0.6093875757405811
# LinearRegression 의 정답률 :  0.609387575740581
# LinearSVR 의 정답률 :  0.5835239782174557
# MLPRegressor 의 정답률 :  0.7025634137333636
# MultiOutputRegressor 안 나온 놈 !!!
# MultiTaskElasticNet 안 나온 놈 !!!
# MultiTaskElasticNetCV 안 나온 놈 !!!
# MultiTaskLasso 안 나온 놈 !!!
# MultiTaskLassoCV 안 나온 놈 !!!
# NuSVR 의 정답률 :  0.6643264874418959
# OrthogonalMatchingPursuit 의 정답률 :  0.47136253068436096
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6016653478988725
# PLSCanonical 안 나온 놈 !!!
# PLSRegression 의 정답률 :  0.5293354337368597
# PassiveAggressiveRegressor 의 정답률 :  0.5606027249622505
# PoissonRegressor 의 정답률 :  0.03909799565359928
