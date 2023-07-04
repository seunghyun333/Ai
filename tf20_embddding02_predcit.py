#predict 
import numpy as np
from keras.preprocessing.text import Tokenizer

#1. 데이터
docs =['재밌어요','재미없다','돈 아깝다','숙면했어요','최고에요','꼭 봐라',
       '세 번 봐라', '또 보고싶다', 'n회차 관람', '배우가 잘 생기긴 했어요', 
       '발연기에요', '추천해요', '최악','후회된다','돈 버렸다','글쎄요','보다 나왔다',
       '망작이다','연기가 어색해요','차라리 기부할걸','다음편 나왔으면 좋겠다',
       '다른 거 볼걸','감동이다']

# 라벨링 긍정 1, 부정 0
labels = np.array([1,0,0,0,1,1,
                   1,1,1,0,
                   0,1,0,0,0,0,0,
                   0,0,0,1,
                   0,1])

#tokenizer
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'돈': 1, '봐라': 2, '재밌어요': 3, '재미없다': 4, '아깝다': 5, '숙면했어요': 6, '최고에요': 7, '꼭': 8, '세': 9, '번': 10, 
# '또': 11, '보고싶다': 12, 'n회차': 13, '관람': 14, '배우가': 15, '잘': 16, '생기긴': 17, '했어요': 18, '발연기에요': 19, 
# '추천해요': 20, '최악': 21, '후회된다': 22, '버렸다': 23, '글쎄요': 24, '보다': 25, '나왔다': 26, '망작이다': 27, '연기가': 28, 
# '어색해요': 29, '차라리': 30, '기부할걸': 31, '다음편': 32, '나왔으면': 33, '좋겠다': 34, '다른': 35, '거': 36, '볼걸': 37, '감동이다': 38}

x= token.texts_to_sequences(docs)
print(x)
# [[3], [4], [1, 5], [6], [7], [8, 2], [9, 10, 2], [11, 12], [13, 14], [15, 16, 17, 18], [19], [20], 
#  [21], [22], [1, 23], [24], [25, 26], [27], [28, 29], [30, 31], [32, 33, 34], [35, 36, 37], [38]]

# pad_sequences
from keras_preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=4 ) #padding = 어디다가 넣을거냐 pre 앞,maxlen 제일 긴거 몇개냐
print(pad_x)
'''
[[ 0  0  0  3]
 [ 0  0  0  4]
...
 '''
print(pad_x.shape)  #(23, 4)  전체 단어 개수:23,  max leng: 4

word_size = len(token.word_index)       #길이 확인: len 
print('word_size 는 ', word_size)   #word_size 는  38, 단어 사전 개수가 38개 

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

model= Sequential()
model.add(Embedding(input_dim= 39, output_dim=5, input_length=4, ))# 총 단어갯수+1, 아웃풋 딤(노드 수(내맘)), 제일 긴 문장의 길이, 파라미터이름안적어도됨
model.add(LSTM(8)) # 시간 순서대로 
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   #이진분류여서 sigmoid
#model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.fit(pad_x, labels, epochs=10, batch_size=1)     # x값: pad_x, y값: labels 

#4 평가 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)

# loss :  0.05028156191110611
# acc :  1.0

###################[predict ]############
#x_predict = '영화가 정말 정말 재미없네' #부정
#x_predict = '정말 정말 재미있고 최고에요'   #긍정
#x_predict = '망작이고 돈아깝고 졸렸음 우엑' #부정
#x_predict = '감동적이고 배우가 잘생기고 좋았음' #부정 [[0.3419319]] ******************
x_predict = '배우가 잘 생기긴 했어요'



#1) tokenizer
token1 = Tokenizer()
x_predict = np.array([x_predict])
print(x_predict)    #['영화가 정말 정말 재미없네 진짜'] => 리스트 안에 들어감 . 그래야 토크나이저, 사용 가능 
token1.fit_on_texts(x_predict)     
x_pred = token1.texts_to_sequences(x_predict)      #순서대로
print(token1.word_index)        #{'정말': 1, '영화가': 2, '재미없네': 3, '진짜': 4}
print(len(token1.word_index))   #4
print(x_pred)       #[[2, 1, 1, 3, 4]]

 
#2) pad_secuences -기능이 뭐지?
x_pred = pad_sequences(x_pred, padding='pre')
print(x_pred.shape)     #(1, 4)
print(x_pred)


#3) model.predict
y_pred = model.predict(x_pred)

if y_pred < 0.5 :
    print("부정")
    
else : print("긍정")

print(y_pred)



# 실습: 
# 1. predict 결과 제대로 출력되도록
# 2. score를 긍정부정으로 출력하기


score = float(model.predict(x_pred))
if y_pred < 0.5:
    print("{:.2f} 확률로 부정 \n".format((1-score)*100))
else :
    print("{:.2f} 확률로 긍정 \n".format((score)*100))

