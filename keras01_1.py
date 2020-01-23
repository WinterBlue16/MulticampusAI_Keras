# question 1
#1. 데이터, 라이브러리 불러오기
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 데이터 형태 확인
# print(x.shape)
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(512, input_dim=1) # 레이어 추가
model.add(Dense(512)) # Node 조절
model.add(Dense(1))

#3. 모델 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침 # 이 때문에 아래서 acc가 나온다.
model.fit(x, y, epochs=420, batch_size=1)

#4. 평가예측 
loss, mse = model.evaluate(x, y, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('mse :', mse)

x_prd = np.array([11, 12, 13])
predictions = model.predict(x_prd)
print(predictions)

x_test = model.predict(x)
print(x_test)
# 원래 회귀모델에서는 acc를 사용하지 않는다.
