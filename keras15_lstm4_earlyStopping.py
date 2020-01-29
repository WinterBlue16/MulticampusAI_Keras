
# LSTM Model 4

from numpy import array

# 1. 데이터 생성하기
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8],
            [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]]) 
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) # 벡터가 아닌 행렬로 집어넣을 경우 shape 오류가 발생한다!!

'''
print(x.shape)
print(y.shape)
'''

#  2. 데이터 reshape
x = x.reshape(x.shape[0], x.shape[1], 1) 
# x = x.reshape(13, 3, 1)

# 3. 모델 만들기 
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()

model.add(LSTM(128, activation='relu', input_shape=(3, 1))) # (None, 3, 1) => (row(=행) 무시, column 수(=열), 몇 개씩 자르는지), 데이터 전처리/하이퍼 파라미터 튜닝에도 해당
model.add(Dense(64))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(1))

#model.summary()


# 5. 모델 훈련시키기(EarlyStopping 추가)
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])

from keras.callbacks import EarlyStopping

early_stoppping = EarlyStopping(monitor='loss', patience=95, mode='auto')# patience=N, 원하는 값이 나온 후(ex> min, max), 다른 값이 나오지 않은 상태로 N번 지날 경우 멈춤 
model.fit(x, y, epochs=800, batch_size=1, callbacks=[early_stoppping])# early_stopping은 리스트 형식으로 넣어야 함 => [early_stopping] 

# verbose=0 => 훈련과정을 보여주지 않음 
# verbose=2 => 훈련 과정에서 막대 제거 
# verbose=3 => epoch만 보여줌 


# 6. 모델 평가예측
loss, mae = model.evaluate(x, y, batch_size=1) # loss는 자동적으로 출력
print(loss, mae)


# 7. 실제 예측값 구하기
x_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]]) 
# loss가 높은데도 2, 3번째 predict값이 정답에 근접할 때가 있다. 반대로 loss가 낮은데 1번째 값만 정답에 가까운 경우가 생긴다. 
x_input = x_input.reshape(x_input.shape[0], x_input.shape[1], 1)

y_predict = model.predict(x_input)
print(y_predict)


# 8. 하이퍼 파라미터 튜닝
# 튜닝 가능한 파라미터 => epochs, batch size, Dense(layer) 