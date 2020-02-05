# Tensorboard

#1. 데이터, 라이브러리 불러오기
import numpy as np

x = np.array([range(1, 101), range(101,201), range(301, 401)]) # (3, 100)

y1 = np.array([range(1, 101)])
y2 = np.array([range(101, 201)])
y3 = np.array([range(301, 401)])

x = x.T
y1 = y1.T
y2 = y2.T
y3 = y3.T

'''
print(x.shape)
print(y.shape) 
'''
x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. test, val, train 데이터 나누기
from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(x, train_size=0.6, shuffle=False) 
x_val, x_test= train_test_split(x_test, test_size=0.5, shuffle=False)
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, train_size=0.6, shuffle=False) 
y1_val, y1_test, y2_val, y2_test, y3_val, y3_test  = train_test_split(y1_test, y2_test, y3_test, test_size=0.5, shuffle=False)

#3. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Input 

# model 1
input1 = Input(shape=(3,1))
LSTM1 = LSTM(256, activation='relu', input_shape=(3, 1), return_sequences=True)(input1) # input_shape = (1,), 벡터가 1개라는 뜻
LSTM1 = LSTM(256, activation='relu', return_sequences=True)(LSTM1)
LSTM1 = LSTM(256, activation='selu', return_sequences=True)(LSTM1)
LSTM1 = LSTM(64, activation='selu')(LSTM1)
output1 = Dense(1)(LSTM1) # regression(=회귀분석) 문제이므로 output은 하나여야 한다. # 열을 기준으로 볼 것!


# 분기 만들기
# 1번째 output model 
output_1 = Dense(64)(output1)
output_1 = Dense(32)(output_1)
output_1 = Dense(1)(output_1)

# 2번째 output model 
output_2 = Dense(128)(output1)
output_2 = Dense(256)(output_2)
output_2 = Dense(1)(output_2)

# 3번째 output model
output_3 = Dense(8)(output1)
output_3 = Dense(4)(output_3)
output_3 = Dense(1)(output_3)

model = Model(inputs=input1, outputs=[output_1, output_2, output_3])

model.summary() # 모델 구조 확인


#3. 모델 훈련
from keras.callbacks import EarlyStopping, TensorBoard

td_hist = TensorBoard(log_dir='./graph',
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True) # cmd 열기 => 작업폴더(keras)로 이동 => 명령어 tensorboard --logdir=./graph

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')# patience=N, 원하는 값이 나온 후(ex> min, max), 다른 값이 나오지 않은 상태로 N번 지날 경우 멈춤 
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.

model.fit(x_train, [y1_train, y2_train, y3_train], epochs=100, batch_size=2, 
          callbacks=[early_stopping, td_hist], 
          validation_data=(x_val, [y1_val, y2_val, y3_val]))


#4. 평가예측 
AAA = model.evaluate(x_test, [y1_test, y2_test, y3_test], batch_size=2) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print(AAA)


# 새로운 데이터 넣어보기
x_prd = np.array([[290, 974, 467], [235, 436, 765], [473, 569, 907]]).T
x_prd = x_prd.reshape(x_prd.shape[0], x_prd.shape[1], 1)
g = model.predict(x_prd, batch_size=2)
print(g)

y_predict = model.predict(x_test, batch_size=5)


# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(x, y): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(x, y))

x = [RMSE(y1_test, y_predict[0]), RMSE(y2_test, y_predict[1]), RMSE(y3_test, y_predict[2])]
RMSE_FINAL = np.mean(x)

print('RMSE :', RMSE_FINAL)


# R2 구하기
# R2 정의 및 참고 : https://newsight.tistory.com/259
from sklearn.metrics import r2_score

y = [r2_score(y1_test, y_predict[0]), r2_score(y2_test, y_predict[1]), r2_score(y3_test, y_predict[2])]
r2_y_predict = np.mean(y) 
print("R2 :", r2_y_predict)


# visual studio code 참고(https://demun.github.io/vscode-tutorial/shortcuts/)