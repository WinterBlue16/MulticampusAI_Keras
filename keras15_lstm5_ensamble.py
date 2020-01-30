
# LSTM Model 5
# LSTM + LSTM ensamble Model

from numpy import array

# 1. 데이터 생성하기
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8],
            [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]]) 
y1 = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) # 벡터가 아닌 행렬로 집어넣을 경우 shape 오류가 발생한다!!

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60], [50, 60, 70], 
            [60, 70, 80],[70, 80, 90], [80, 90, 100], [90, 100, 110], [100, 110, 120], 
            [2, 3, 4], [3, 4, 5], [4, 5, 6]]) 
y2 = array([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 5, 6, 7])
# y2 = array([40, 50, 60, 70, 80, 90, 100, 110, 5, 6, 7]) # ensamble Model은 input이 다수일 경우 shape이 모두 같아야 한다!

'''
print(x.shape)
print(y.shape)
'''

#  2. 데이터 reshape
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) 
 
# x = x.reshape(13, 3, 1)

# 3. 모델 만들기 
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input

# LSTM 1
input1 = Input(shape=(3, 1))
dense1 = LSTM(128, activation='relu')(input1)
dense1 = Dense(64)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
output_1 = Dense(4)(dense1)

# LSTM 2
input2 = Input(shape=(3, 1))
dense2 = LSTM(128, activation='relu')(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(256)(dense2)
dense2 = Dense(128)(dense2)
dense2 = Dense(256)(dense2)
dense2 = Dense(256)(dense2)
output_2 = Dense(4)(dense2)

# Merge
from keras.layers.merge import concatenate, Add, Multiply, Minimum, Maximum, Average, Subtract, Dot # model을 사슬처럼 엮다.
merge1 = concatenate([output_1, output_2])
# merge1 = Add()([output_1, output_2])
# merge1 = Multiply()([output_1, output_2])
# merge1 = Minimum()([output_1, output_2])
# merge1 = Maximum()([output_1, output_2])
# merge1 = Average()([output_1, output_2])
# merge1 = Subtract()([output_1, output_2])

# output 1(분기1)
output1 = Dense(64)(merge1)
output1 = Dense(256)(output1)
output1 = Dense(8)(output1)
output1 = Dense(1)(output1)


# output 2(분기2)
output2 = Dense(32)(merge1)
output2 = Dense(128)(output2)
output2 = Dense(64)(output2)
output2 = Dense(1)(output2)


# 4 .함수 선언
model = Model(inputs = [input1, input2],
              outputs = [output1, output2])

model.summary()


# 5. 모델 훈련시키기(EarlyStopping 추가)
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])

from keras.callbacks import EarlyStopping

early_stoppping = EarlyStopping(monitor='loss', patience=90, mode='auto')# patience=N, 원하는 값이 나온 후(ex> min, max), 다른 값이 나오지 않은 상태로 N번 지날 경우 멈춤 
model.fit([x1, x2], [y1, y2], epochs=1000, batch_size=1, callbacks=[early_stoppping])# early_stopping은 리스트 형식으로 넣어야 함 => [early_stopping] 


# 6. 모델 평가예측
aaa = model.evaluate([x1, x2], [y1, y2], batch_size=1) # loss는 자동적으로 출력
print(aaa)


# 7. 실제 예측값 구하기
x1_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]]) 
x2_input = array([[60.5, 70.5, 80.5], [500, 600, 700], [700, 800, 900], [160, 170, 180]]) 


x1_input = x1_input.reshape(x1_input.shape[0], x1_input.shape[1], 1)
x2_input = x2_input.reshape(x2_input.shape[0], x2_input.shape[1], 1)
y_predict = model.predict([x1_input, x2_input])

print(y_predict)

# 8. 하이퍼 파라미터 튜닝
# 튜닝 가능한 파라미터 => epochs, batch size, Dense(layer) 수, node 수, patience 