import numpy as np

oliers = np.array([-50,-10,2,3,4,5,6,7,8,9,10,11,12,50])
print(oliers.shape)     #(14,)
oliers = oliers.reshape(-1,1)
print(oliers.shape)     #(14, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination =.1) #이상치 몇 퍼 지정할지 정하기

outliers.fit(oliers)
result = outliers.predict(oliers)

print(result)
print(result.shape) #(14,)
#[-1  1  1  1  1  1  1  1  1  1  1  1  1 -1]v # 패딩ㅇ으로 하나씨기 더 잫음듯 