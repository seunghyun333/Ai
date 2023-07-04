# 파이썬 기초
# 리스트

a = [38, 21, 53, 62, 19]
print(a[0])
print(a[1])

# 리스트 문자 출력
b = ['메이킷', '우진', '시은']
print(b)
print(b[0])
print(b[1])
print(b[2])

# 리스트 정수와 문자 출력
c = ['james', 26, 175.3, True]
print(c)

d = ['메이킷', '우진', '제임스', '시은']
# print([d[0], d[1]])
# print([d[1], d[2], d[3]])
# print([d[2], d[3]])
# print(d)

# 리스트를 출력하기 
print(d[0:2])   #2를 넣으면 2앞까지 나옴. 따라서 0부터 1까지, 0을 안 넣으면, 그러니까 아무것도 안 쓰면 0번째부터출력력
print(d[1:4])
print(d[2:4])
print(d[0:4])

# extend() 함수사용하여 리스트 이어붙이기
e = ['우진', '시은']
f = ['메이킷', '소피아', '하워드']
e.extend(f)      #e뒤에 f를 붙이겠다 , 두개의 데이터(리스트)를 이을 때 사용, append랑 다름, append는 통채로 붙임
print(e)
print(f)

f.extend(e)  #extend로 리스트를 이음
print(f[3])  