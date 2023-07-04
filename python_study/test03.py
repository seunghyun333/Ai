import numpy as np

a = np.array([[1, 2 ,3],[4, 5,6]]) # 리스트가 두개면 리스트를 리스트로 묶어줘야함
print("Original: \n", a)  #\n 줄바꿈 \t 띄어쓰기기

a_transpose = np.transpose(a)
print("Transpose :\n", a_transpose)
'''
 [[1 4]
 [2 5]
 [3 6]]
 '''

b = np.array([[1,2,3],[4,5,6]])
print("Original: \n", a)

b_reshape=np.reshape(b, (3,2))
print("reshape :\n", b_reshape)
'''
reshape :
 [[1 2]
 [3 4]
 [5 6]]
'''
#transpose()와 reshape()의 차이 : trans는 두개의 리스트 순서대로 하나씩 짝 지어주고 
# reshape은 두개 리스트를 하나의 리스트로 합치고 (3,2)면 리스트 세개를 2개씩 ,,? ?? 
#시간에 달린 데이터는 transpose 이용 