
# 0. 라이브러리 불러오기
import numpy as np
from numpy import array
import pandas as pd

# 1. 데이터 불러오기
data_samsung = pd.read_csv('./SAMSUNG/samsung.csv', encoding='cp949', thousands=',')
copy1 = data_samsung.copy()
copy2 = data_samsung.copy()
data_kospi = pd.read_csv('./SAMSUNG/samsung.csv', encoding='cp949',thousands=',')
copy3 = data_kospi.copy()

data_samsung.sort_values(by=['일자'], ascending=True, inplace=True, axis=0) # 일자 오름차순 정렬
data_kospi.sort_values(by=['일자'], ascending=True, inplace=True, axis=0) 


print(data_samsung.head(20)) #(426, 6)
print(data_kospi.tail(10)) #(426, 6)

# print(data_samsung.tail(20)) #(426, 6)
# print(data_kospi.tail(10)) #(426, 6)

print(data_samsung.info()) # null 값 없음


# 2. 데이터 전처리
# 필요한 열만 추출
data_x = data_samsung[['시가', '고가', '저가', '종가', '거래량']]
data_x = np.array(data_x) # array로 변환 # data_x.values도 가능!

x_prd = data_x[423:] # 미리 predict 값 지정
print(x_prd.shape)

def split_sequence(sequence, n_steps):
    X = list()
    for i in range(len(sequence)): 
        end_ix = i + n_steps       
        if end_ix > len(sequence)-1: # 수정
            break
        seq_x = sequence[i:end_ix, :] # 수정
        X.append(seq_x) # X 리스트에 추가 
    return array(X) # numpy 배열로 만들기

n_steps = 3
x = split_sequence(data_x, n_steps)
print('n_steps :', n_steps)
print(x)
print(x.shape)


y = copy2['종가']
y = y[3:]
print(y.shape)

# data_x = copy1.values # numpy 배열로 변환
y = y.values

print(type(x))
print(type(y))


# 3. split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
# print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], -1)
scaler = MinMaxScaler() 
scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 적용

# print(x_test.shape)
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = scaler.transform(x_test) # 적용

print(x_test.shape)
print(x_train.shape)


# 4. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Dense(256, input_shape=(15,)))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dense(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(1))

# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)


# 5. 모델 훈련
# from keras.callbacks import EarlyStopping, TensorBoard

# early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mae', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.15)


# 6. 평가예측 
loss, mae = model.evaluate(x_test, y_test, batch_size=5) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산
print('loss :', loss)
print('mae :', mae)

# model.summary()


# 7. RMSE 구하기
from sklearn.metrics import mean_squared_error

y_predict = model.predict(x_test, batch_size=5)

def RMSE(y_test, y_predict): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))


# 9. 예측값 구하기(2월 3일)
x_prd = x_prd.reshape(1, 15)
x_prd = scaler.transform(x_prd)
result = model.predict(x_prd, batch_size=5)
print(result)

