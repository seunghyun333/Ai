import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6,8,10],
                    [2,4,np.nan, 8,np.nan,],
                    [2,4,6,8,10],
                    [np.nan,4, np.nan, 8, np.nan]])
# print(data)
# print(data.shape)
#      0    1    2  3     4
# 0  2.0  NaN  6.0  8  10.0
# 1  2.0  4.0  NaN  8   NaN
# 2  2.0  4.0  6.0  8  10.0
# 3  NaN  4.0  NaN  8   NaN
# (4, 5)

data = data.transpose()
data.columns = ['x1','x2','x3', 'x4']

# print(data)
# print(data.shape)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# (5, 4)

#결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

# 1. 결측치 삭제 
print(data.dropna(axis=0)) #난 값이 들어있는 컬럼이 다 없어짐 
print(data.shape)
'''
# print(data.dropna()) 
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
# (5, 4)

# axis=0이면 컬럼 다 살아있는데 행이 사라짐 
    x1   x2   x3   x4
3  8.0  8.0  8.0  8.0
(5, 4)

# axis=1이면 컬럼 열이 다 사라짐 
     x3
0   2.0
1   4.0
2   6.0
3   8.0
4  10.0
(5, 4)
'''
'''
#2. 특정값으로 대체 
means = data.mean()     # 평균 
median = data.median()  #중간 값 

data2 = data.fillna(means)
print(data2)
     x1        x2    x3   x4
0   2.0  2.000000   2.0  6.0
1   6.5  4.000000   4.0  4.0
2   6.0  4.666667   6.0  6.0
3   8.0  8.000000   8.0  8.0
4  10.0  4.666667  10.0  6.0

data3= data.fillna(median)
print(data3)
     x1   x2    x3   x4
0   2.0  2.0   2.0  6.0
1   7.0  4.0   4.0  4.0
2   6.0  4.0   6.0  6.0
3   8.0  8.0   8.0  8.0
4  10.0  4.0  10.0  6.0
'''