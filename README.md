# Ai

# AI 개념 정리

1일차 
#1. 인공지능 개념 정리 – 머신러닝, 딥러닝
#2. 퍼셉트론 (Perceptron)
#3. 다층 퍼셉트론 (Multi-Layer Perceptron: MLP)
#4. 옵티마이저 (Optimizer)
#5. 학습률 (learning rate)
#6. 경사하강법 (Gradient Descent)
#7. 손실함수 (Loss Function)
#8. 활성화 함수 (Activation Function) – Sigmoid, ReLU, Softmax

2일차
#1. train_test_split
#2. validation split
#3. matplotli b – scatter (산점도 )
#4. R2 score ( 결정계수 )
#5. 회귀분석 
#6. 분류분석
#7. 이진분류
#8. 다중분류
#9. 원 핫 인코딩 (One Hot Encoding)
#10. 난수값 (random_state)

3일차
#1. 정확도 (accuracy score)
#2. 과적합 (overfiting)
#3. 콜백(callback)
#4. optimizer: 손실 함수의 값을 최소화하기 위해 모델의 가중치를 조정하는 알고리즘 

4일차
#1. 이미지 분석 - CNN 모델
  CNN(합성곱 신경망): convolution - 3차원 가로 세로
  FC(완전연결 신경망): fully connected - 모든 뉴런이 다 연결됨

5일차
자연어처리(NLP) Natural Language Process
워드 임베딩(word Embedding)
classic : 언어 별로 나누어서 pre-processing 들어감
deep learning: 먼저 prepreocessing (=숫자변환)

6일차 머신러닝
SVM 모델 (suport vecter machine)
  Decision Tree
  Random Forest

[Boosting 계열의 모델] 
  Adaptive Boosting (AdaBoost) : 다수결을 통한 정답 분류 및 오답에 가중치 부여
  Gradient Boosting Mode (GBM) : LightGBM, CatBoost, XGBoost - Gradient Boositng
  K-Fold : ML 모델에서 가장 보편적으로 사용되는 교차 검증 기법

[스케일링 (Scaling)]
  Normalization(정규화) : 데이터를 0과 1사이의 범위로 조정, 상대적인 크기를 유지하면서 모델이 더 잘 학습하게 도움
  Standardization(표준화) : 평균이 0, 표준편차가 1로 변환, 데이터의 분포를 조정하여 이상치에 대해 덜 민감하게 만듬
  MinMaxScaler(), 0과 1사이로 스케일링, 이상치에 민감하며, 회귀에 유용
  StandardScaler(), 평균을 0, 분산을 1로 스케일링, 이상치에 민감하며, 분류에 유용함
  MaxAbsScaler(), 특성의 절대값이 0과 1 사이가 되도록 스케일링. 모든 값은 -1과 1사이로 표현되며 양수일 경우 minmax와 동일 
  RobustScaler(), 평균과 분산 대신 중간 값과 사분위 값을 사용, 이상치 영향을 최소화할 수 있음 

[하이퍼파라미터튜닝]
  -그리드서치
  -아웃라이어
  -배깅(Bagging)
  -보팅(voting)

