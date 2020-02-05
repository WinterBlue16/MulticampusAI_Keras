# question 1
#1. 데이터, 라이브러리 불러오기
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_val = np.array([101, 102, 103, 104, 105]) # 원래는 train set에서 나눠준다.
y_val = np.array([101, 102, 103, 104, 105])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])


# 데이터 형태 확인
# print(x.shape)
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

# model.add(Dense(512, input_dim=1)) # 레이어 추가
model.add(Dense(512, input_shape=(1,))) # input_shape = (1,), 벡터가 1개라는 뜻
model.add(Dense(256)) # Node 조절 
model.add(Dense(256)) 
model.add(Dense(1))

# model.summary()
# (None, 512)는 행렬을 표현한 것

#3. 모델 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침 # 이 때문에 아래서 acc가 나온다.
model.fit(x_train, y_train, epochs=420, batch_size=1, validation_data=(x_val, y_val))

#4. 평가예측 
loss, mse = model.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('mse :', mse)

x_prd = np.array([11, 12, 13])
predictions = model.predict(x_prd)
print(predictions)

# x_test = model.predict(x)
# print(x_test)
# 원래 회귀모델에서는 acc를 사용하지 않는다.