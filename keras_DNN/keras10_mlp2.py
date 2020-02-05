# Multi Layer Perceptron
# parameter 조정으로 정확도 상승, 저하 실험

#1. 데이터, 라이브러리 불러오기
import numpy as np

x = np.array([range(1, 101), range(101,201), range(301, 401)]) # (3, 100)
y = np.array([range(1, 101)]) # (1, 100) # 대괄호가 있을 경우 벡터가 아닌 행렬이 된다!
# y = np.array(range(1, 101)) # (100,) # Output은 벡터여도 전혀 상관없다. reshape도 필요없음!

# print(x.shape) # (3, 100)
# print(y.shape) # (1, 100)
# print(y.shape) # (100,)

# 행, 열 변환(reshape, np.transpose, T)
# 참고 : https://rfriend.tistory.com/289
x = x.T # reshape(x, [10,2]) # x = np.transpose(x)
y = y.T # reshape(y, [10,2]) # y = np.transpose(y)


# 2. test, val, train 데이터 나누기
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False) # 전체 데이터에서 train, test 분리
# 입력값은 x(x_train과 x_test로 분리), y(y_train과 y_test로 분리)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False) # test 데이터의 절반을 validation로 분리


#3. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 

model=Sequential()# 순차적으로 실행

# model.add(Dense(512, input_dim=3)) # 레이어 추가
model.add(Dense(256, input_shape=(3,))) # input_shape = (1,), 벡터가 1개라는 뜻
model.add(Dense(256)) # Node 값 조절 
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256)) 
model.add(Dense(1)) # regression(=회귀분석) 문제이므로 output은 하나여야 한다. # 열을 기준으로 볼 것!

# model.summary() # 모델 구조 확인


#3. 모델 훈련
model.compile(loss='mae', optimizer='adam', 
              metrics=['mae']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_val, y_val))


#4. 평가예측 
loss, mae = model.evaluate(x_test, y_test, batch_size=100) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('mae :', mae)


# 새로운 데이터 넣어보기
x_prd = np.array([[290, 974, 467], [235, 436, 765], [473, 569, 907]]).T
g = model.predict(x_prd, batch_size=100)
print(g)

y_predict = model.predict(x_test, batch_size=5)


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