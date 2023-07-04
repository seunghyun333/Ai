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
 [ 0  0  1  5]
 [ 0  0  0  6]
 [ 0  0  0  7]
 [ 0  0  8  2]
 [ 0  9 10  2]
 [ 0  0 11 12]
 [ 0  0 13 14]
 [15 16 17 18]
 [ 0  0  0 19]
 [ 0  0  0 20]
 [ 0  0  0 21]
 [ 0  0  0 22]
 [ 0  0  1 23]
 [ 0  0  0 24]
 [ 0  0 25 26]
 [ 0  0  0 27]
 [ 0  0 28 29]
 [ 0  0 30 31]
 [ 0 32 33 34]
 [ 0 35 36 37]
 [ 0  0  0 38]]
 '''
print(pad_x.shape)  #(23, 4)  전체 단어 개수:23,  max leng: 4

word_size = len(token.word_index)       #길이 확인: len 
print('word_size 는 ', word_size)   #word_size 는  38, 단어 사전 개수가 38개 

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

model= Sequential()
model.add(Embedding(input_dim= 39, output_dim=10, input_length=4, ))# 총 단어갯수+1, 아웃풋 딤(노드 수(내맘)), 제일 긴 문장의 길이, 파라미터이름안적어도됨
model.add(LSTM(32)) # 시간 순서대로 
model.add(Dense(1, activation='sigmoid'))   #이진분류여서 sigmoid

#model.summary()

'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 4, 10)             390                  3차원

 lstm (LSTM)                 (None, 32)                5504                 3차원을 받아서 2차원으로 내보냄, 따라서 reshape, flatten 안 해도 됨

 dense (Dense)               (None, 1)                 33

=================================================================
'''

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.fit(pad_x, labels, epochs=100, batch_size=16)     # x값: pad_x, y값: labels 

#4 평가 예측

loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)

# loss :  0.05028156191110611
# acc :  1.0

