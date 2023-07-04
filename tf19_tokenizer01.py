##tokenizer


import numpy as np
from keras.preprocessing.text import Tokenizer

text= '나는 진짜 매우 매우 매우 매우 매우 많이 많이 맛있는 밥을 엄청 엄청 엄청 먹어서 매우 배가 부르다.'

token = Tokenizer()
token.fit_on_texts([text]) # fit_on 하면서 index 생성 
index = token.word_index

print(token.word_index)
#{'매우': 1, '엄청': 2, '많이': 3, '나는': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '먹어서': 8, '배가': 9, '부르다': 10} #빈도수로 순서가 생성됨 

x = token.texts_to_sequences([text])
print(x)
#[[4, 5, 1, 1, 1, 1, 1, 3, 3, 6, 7, 2, 2, 2, 8, 1, 9, 10]]

#[to_categorical]
from keras.utils import to_categorical

x = to_categorical(x)   #onehotencoding하면 index+1개로 만들어짐 
print(x)
'''
패딩 넣어줘서 글씨 10갠데 11줄임 
[[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
  '''
print(x.shape)  #(1, 18, 11) 1은 onehot하면 3차원으로 받아옴, 18은 전체 단어, 11개는 뽑아낸 중복단어 



'''
#LSTM : 순서대로 들어가서 데이터들이 분석됨 순서가 중요하기 때문에 LSTM 방식으로 해줘야함
############ OneHotEngoder 수정 ###############
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
x = x.reshape(-1, 11, 9)
onehot_encoder.fit(x)
x= onehot_encoder.transform(x)
print(x)
print(x.shape)

# 에러 : AttributeError: 'list' object has no attribute 'reshape'
'''






