# question 1
#1. 데이터, 라이브러리 불러오기
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(1, 101)) 

from sklearn.model_selection import train_test_split

# 2. test, val, train 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False) # 전체 데이터에서 train, test 분리
# 입력값은 x(x_train과 x_test로 분리), y(y_train과 y_test로 분리)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False) # test 데이터의 절반을 val로 분리

#3. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 

model=Sequential()# 순차적으로 실행

# model.add(Dense(512, input_dim=1)) # 레이어 추가
model.add(Dense(512, input_shape=(1,))) # input_shape = (1,), 벡터가 1개라는 뜻
model.add(Dense(256)) # Node 값 조절 
model.add(Dense(256)) 
model.add(Dense(1)) # regression(=회귀분석) 문제이므로 output은 하나여야 한다.

# model.summary() # 모델 구조 확인

#3. 모델 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_val, y_val))

#4. 평가예측 
loss, mse = model.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('mse :', mse)

x_prd = np.array([290, 154, 467])
g = model.predict(x_prd)
print(g)

y_predict = model.predict(x_test, batch_size=1)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
# R2 정의 및 참고 : https://newsight.tistory.com/259
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)

# visual studio code 참고(https://demun.github.io/vscode-tutorial/shortcuts/)