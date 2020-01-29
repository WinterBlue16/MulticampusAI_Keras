# RNN(Recurrent Neural Networks) => LSTM Model

# 1. 라이브러리 불러오기
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 2. 데이터 생성
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5],  [4, 5, 6], [7, 8, 9]]) 
y = array([4, 5, 6, 7, 10]) 

'''
print(x.shape[0]) # 행
print(x.shape[1]) # 열

print(x.shape) # (5, 3)
print(y.shape) # (5,)
'''

# 3. 데이터 reshape
x = x.reshape(x.shape[0], x.shape[1], 1) # 가급적이면 요런 표현이 좋음
# x = x.reshape(5, 3, 1)


# 4. 모델 만들기 
model = Sequential()

model.add(LSTM(128, activation='relu', input_shape=(3, 1))) # (None, 3, 1) => (row(=행) 무시, column 수(=열), 몇 개씩 자르는지), 데이터 전처리/하이퍼 파라미터 튜닝에도 해당
model.add(Dense(64))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(1))

model.summary()


# 5. 모델 훈련시키기
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x, y, epochs=190, batch_size=1)


# 6. 모델 평가예측
loss, mae = model.evaluate(x, y, batch_size=1) # loss는 자동적으로 출력
print(loss, mae)


# 7. 실제 예측값 구하기
x_input = array([6, 7, 8])
x_input = x_input.reshape(1, 3, 1)

y_predict = model.predict(x_input)
print(y_predict)

# 8. 하이퍼 파라미터 튜닝
# 튜닝 가능한 파라미터 => epochs, batch size, Dense(layer) 