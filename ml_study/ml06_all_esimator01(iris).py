import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore') #경고창 끄기 

#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7, random_state=123, shuffle=True)
from sklearn.metrics import accuracy_score

#스케일러 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
allAlgorithms = all_estimators(type_filter='classifier') #모든 모델 중 분류 모델만 
print('all Algorightms : ', allAlgorithms)
print('몇개', len(allAlgorithms))  # 몇개 41

'''
[('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),
 ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), 
 ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), 
 ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), 
 ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>),
 ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), 
 ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>),
 ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>),
 ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), 
 ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), 
 ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>),
 ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), 
 ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>),
 ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), 
 ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>),
 ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), 
 ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), 
 ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>),
 ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), ('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>),
 ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>),
 ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), 
 ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), 
 ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), 
 ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), 
 ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>),
 ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), 
 ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>),
 ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), 
 ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), 
 ('SGDClassifier',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), ('SVC', <class 'sklearn.svm._classes.SVC'>), (
'''
#3. 출력(평가,예측)
for(name, algorithm) in allAlgorithms:
    try : 
        model = algorithm()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 :', acc )
    except :
        print(name,'안 나온 놈!!')
  