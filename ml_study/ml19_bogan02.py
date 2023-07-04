import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6,8,10],
                    [2,4,np.nan, 8,np.nan,],
                    [2,4,6,8,10],
                    [np.nan,4, np.nan, 8, np.nan]])

data = data.transpose()
data.columns = ['x1','x2','x3', 'x4']

# from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

# imputer = SimpleImputer()   #평균값으로 대체(default)
# imputer = SimpleImputer(strategy='mean') #평균값 
# imputer = SimpleImputer(strategy='median') # 중간값 
# imputer = SimpleImputer(strategy='most_frequent')   #가장 많이 사용된 값 
imputer = SimpleImputer(strategy='constant', fill_value=777) #상수 입력값, default = 0 
imputer.fit(data)
data2 = imputer.transform(data)
print(data2)
'''
imputer = SimpleImputer()
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
 
 imputer = SimpleImputer(strategy='median') # 중간값
 [[ 2.  2.  2.  6.]
 [ 7.  4.  4.  4.]
 [ 6.  4.  6.  6.]
 [ 8.  8.  8.  8.]
 [10.  4. 10.  6.]]
 
 imputer = SimpleImputer(strategy='most_frequent')
 [[ 2.  2.  2.  4.]
 [ 2.  4.  4.  4.]
 [ 6.  2.  6.  4.]
 [ 8.  8.  8.  8.]
 [10.  2. 10.  4.]]
 
 imputer = SimpleImputer(strategy='constant', fill_value=777) 
 [[  2.   2.   2. 777.]
 [777.   4.   4.   4.]
 [  6. 777.   6. 777.]
 [  8.   8.   8.   8.]
 [ 10. 777.  10. 777.]]
 '''