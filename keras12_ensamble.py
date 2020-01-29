# Multi Layer Perceptron
# Ensamble Model 만들기

# model을 단순화하는 것도 가능하지만, 각 model의 depth나 노드 수를 다르게 하여 weight의 값 변화를 측정, 비교하는 것이 가능함.


#1. 데이터, 라이브러리 불러오기
import numpy as np

x1 = np.array([range(1, 101), range(101,201), range(301, 401)]) # (3, 100)
x2 = np.array([range(1001, 1101), range(1101,1201), range(1301, 1401)]) 
y = np.array([range(1, 101)]) # (1, 100) # 대괄호가 있을 경우 벡터가 아닌 행렬이 된다!
# y2 = np.array([range(1001, 1101)]) # input과 output의 갯수는 다를 수 있어도 행은 동일해야 한다!

# 행, 열 변환(reshape, np.transpose, T)
# 참고 : https://rfriend.tistory.com/289
x1 = x1.T # reshape(x, [10,2]) # x = np.transpose(x)
x2 = x2.T # reshape(x, [10,2]) # x = np.transpose(x)
y = y.T # reshape(y, [10,2]) # y = np.transpose(y)


# 2. test, val, train 데이터 나누기
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.6) # 전체 데이터에서 train, test 분리(꼭 x, y 두 개만 들어갈 필요는 없음!!)
# 입력값은 x(x_train과 x_test로 분리), y(y_train과 y_test로 분리)
x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(x1_test, x2_test, y_test, train_size=0.5) # test 데이터의 절반을 validation로 분리

# print(x1_train.shape)
# print(x1_test.shape)
# print(x1_val.shape)


#3. 모델구성(함수형 model)
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

# model=Sequential()# 순차적으로 실행

# Model 1
input1 = Input(shape=(3,)) # input layer 형태
dense1 = Dense(256)(input1) # 바로 앞 layer를 뒷부분에 명시
dense2 = Dense(64)(dense1)
output1 = Dense(1)(dense2)

# Model 2
input2 = Input(shape=(3,)) 
dense21 = Dense(256)(input2) 
dense22 = Dense(8)(dense21)
dense23 = Dense(128)(dense22)
output2 = Dense(10)(dense23)

# concatenate
# from keras.layers.merge import concatenate # model을 사슬처럼 엮다.
# merge1 = concatenate([output1, output2]) # list 형식(=[]) # merge layer 역시 hidden layer! 따라서 끝 노드 수를 굳이 맞춰주지 않아도 된다.

# Concatenate(참조: https://keras.io/layers/merge/#concatenate)
from keras.layers.merge import Concatenate # from keras.layers import Concatenate
merge1 = Concatenate()([output1, output2])

# Model 3
middle1 = Dense(3)(merge1)
middle2 = Dense(256)(middle1)
output = Dense(1)(middle2)

# 함수형 model 선언
model = Model(inputs = [input1, input2], outputs = output) # input, output이 다수일 경우 list 형식으로 삽입!

# # model.add(Dense(512, input_dim=3)) # 레이어 추가
# model.add(Dense(256, input_shape=(3,))) # input_shape = (1,), 벡터가 1개라는 뜻
# model.add(Dense(256)) # Node 값 조절 
# model.add(Dense(1)) # regression(=회귀분석) 문제이므로 output은 하나여야 한다. # 열을 기준으로 볼 것!

model.summary() # 모델 구조 확인


#3. 모델 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침. # metrics는 보여주는 것.
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1, validation_data=([x1_val, x2_val], y_val))


#4. 평가예측 
loss, mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('mse :', mse)


# 새로운 데이터 넣어보기
x1_prd = np.array([[501, 502, 503], [601, 602, 603], [701, 702, 703]]).T
x2_prd = np.array([[657, 325, 894], [987, 899, 765], [473, 569, 907]]).T
g = model.predict([x1_prd, x2_prd], batch_size=1)
print(g)

# RMSE, R2 확인을 위한 변수 생성
y_predict = model.predict([x1_test, x2_test], batch_size=1)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 실제 정답값(=y_test), 모델을 통한 예측값(=y_predict) 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))


# R2 구하기
# R2 정의 및 참고 : https://newsight.tistory.com/259
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)


# visual studio code 참고(https://demun.github.io/vscode-tutorial/shortcuts/)