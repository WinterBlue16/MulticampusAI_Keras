import numpy as np
import pandas as pd

# 3. 저장한 np.array 파일 불러오기
samsung = np.load('./SAMSUNG/DATA/Samsung.npy')
kospi200 = np.load('./SAMSUNG/DATA/kospi200.npy')

# print(samsung)
# print(samsung.shape)
print(kospi200)
print(kospi200.shape)

def split_xy5(dataset, time_steps, y_column): # y_column 뒤에 다른 column을 추가하여 dataset을 분리하는 것도 가능하다(target이 2개 이상일 경우)
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps # x의 끝 값(몇 개씩 자를 것인가)
        y_end_number = x_end_number + y_column # y의 시작값
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] # x의 위치 => 행 지정, 열 전부
        tmp_y = dataset[x_end_number:y_end_number, 3] # y의 위치, 숫자는 '종가'(target)의 idx
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x), np.array(y)

time_steps = 25
x, y =split_xy5(samsung, 25, 1)
print(x.shape)
print(y.shape)
print(x[0,:], '\n', y[0])


# 3. 데이터 split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1, shuffle=False)

# print(x_train.shape)
# print(x_test.shape)


# 4. 데이터 전처리(3차원 => 2차원)
from sklearn.preprocessing import StandardScaler

x_train = x_train.reshape(x_train.shape[0], -1) # 2차원으로 변경
# x_test = x_test.reshape(x_test.shape[0], x_train.shape[1] * x_train.shape[2]) # 이렇게 써도 가능 # 작은 데이터일 경우 전체 데이터 적용도 ok!
scaler = StandardScaler() # 데이터값 - 평균 / 표준편차
scaler.fit(x_train) 
x_train_scaled = scaler.transform(x_train)  

x_test = x_test.reshape(x_test.shape[0], -1) # 2차원으로 변경
x_test_scaled = scaler.transform(x_test)

print(x_train)
print(x_test)
print(x_train.shape)
print(x_test.shape)

x_train_scaled = x_train.reshape(x_train.shape[0], 25, 5) # 다시 3차원 변경(LSTM 사용 시)
x_test_scaled = x_test.reshape(x_test.shape[0], 25, 5) 

print(x_train_scaled.shape)
print(x_test_scaled.shape)


# 3. 모델 구성
from keras.models import Model
from keras.layers import Input, Dropout, BatchNormalization, Dense, LSTM

input1 = Input(shape=(25, 5))
x = LSTM(256, activation='selu', return_sequences=True)(input1)
x = LSTM(256, activation='selu')(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Dense(256)(x)
x = Dense(256)(x)
x = Dropout(0.3)(x)
output1 = Dense(1)(x)

model = Model(inputs=input1, outputs=output1)


# 5. 모델 훈련
# from keras.callbacks import EarlyStopping, TensorBoard
# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)

# early_stopping = EarlyStopping(monitor='loss', patience=60, mode='auto')


model.compile(loss='mae', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x_train_scaled, y_train, epochs=150, batch_size=10, validation_split=0.2)

# 6. 평가예측 
loss, mae = model.evaluate(x_test_scaled, y_test, batch_size=10) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산
print('loss :', loss)
print('mae :', mae)

# model.summary()


# 7. RMSE 구하기
from sklearn.metrics import mean_squared_error

y_predict = model.predict(x_test_scaled, batch_size=10)

def RMSE(y_test, y_predict): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))


# 9. 예측값 구하기(2월 3일)
x_prd = samsung[401:]
x_prd = x_prd.reshape(1, 125)
x_prd_scaled = scaler.transform(x_prd)
x_prd_scaled = x_prd_scaled.reshape(1, 25, 5)
result = model.predict(x_prd_scaled, batch_size=10)
print(result)

