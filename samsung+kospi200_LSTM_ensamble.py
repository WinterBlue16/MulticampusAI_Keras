# 0. 라이브러리 불러오기
import numpy as np
from numpy import array
import pandas as pd

# 1. 데이터 불러오기
data_samsung = pd.read_csv('./SAMSUNG/samsung.csv', encoding='cp949', thousands=',')
copy1 = data_samsung.copy()
copy2 = data_samsung.copy()
data_kospi = pd.read_csv('./SAMSUNG/kospi200.csv', encoding='cp949',thousands=',')
copy3 = data_kospi.copy()

data_samsung.sort_values(by=['일자'], ascending=True, inplace=True, axis=0) # 일자 오름차순 정렬
data_kospi.sort_values(by=['일자'], ascending=True, inplace=True, axis=0) 


print(data_samsung.head(20)) #(426, 6)
print(data_kospi.head(10)) #(426, 6)

# print(data_samsung.info()) # null 값 없음
print(data_kospi.info()) #(426, 6)
      

# 2. 데이터 전처리
# samsung
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


# kospi200
data_X = data_kospi[['시가', '고가', '저가', '현재가', '거래량']]
data_X = np.array(data_X) # array로 변환 # data_x.values도 가능!
X_prd = data_X[423:] # 미리 predict 값 지정
print(X_prd.shape)


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
X = split_sequence(data_X, n_steps)
print('n_steps :', n_steps)
print(X)
print(X.shape)


# 3. split
from sklearn.model_selection import train_test_split
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, x, y, train_size=0.7)
X_val, X_test, x_val, x_test, y_val, y_test = train_test_split(X_test, x_test, y_test, test_size=0.5)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

# print(x_train.shape)
X_train = X_train.reshape(x_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

scaler = MinMaxScaler() 
scaler.fit(X_train) # 훈련
X_train = scaler.transform(X_train) # 적용
X_test = scaler.transform(X_test)

X_train = X_train.reshape(x_train.shape[0], 3, 5)
X_test = X_test.reshape(x_test.shape[0], 3, 5)


# print(x_test.shape)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 적용
x_test = scaler.transform(x_test) # 적용

x_train = x_train.reshape(x_train.shape[0], 3, 5)
x_test = x_test.reshape(x_test.shape[0], 3, 5)

print(X_test.shape)
print(X_train.shape)
print(x_test.shape)
print(x_train.shape)


# 4. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization

# model 1

input1 = Input(shape=(3, 5))
x1 = LSTM(256, activation='relu', return_sequences=True)(input1)
x1 = LSTM(256, activation='relu')(x1)
x1 = BatchNormalization()(x1)
x1 = Dense(32)(x1)
x1 = Dense(256)(x1)
x1 = Dropout(0.4)(x1)
x1 = Dense(256)(x1)
output1 = Dense(8)(x1)

# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)


# model 2

input2 = Input(shape=(3, 5)) 
x2 = LSTM(256, activation='relu', return_sequences=True)(input2)
x1 = LSTM(256, activation='relu', return_sequences=True)(x2)
x2 = LSTM(256, activation='relu')(x2)
x2 = Dense(256)(x2)
x2 = Dropout(0.4)(x2)
x2 = Dense(256)(x2)
output2 = Dense(8)(x2)

# model 3
from keras.layers.merge import Concatenate # from keras.layers import Concatenate
merge1 = Concatenate()([output1, output2])

# 2번째 output model 
output_1 = Dense(128)(merge1)
output_1 = Dense(256)(output_1)
output_1 = Dense(1)(output_1)

# 함수형 model 선언
model = Model(inputs = [input1, input2], outputs = output_1)

# 5. 모델 훈련
# from keras.callbacks import EarlyStopping, TensorBoard

# early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mae', optimizer='adam', 
              metrics=['mse']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit([X_train, x_train], y_train, epochs=100, batch_size=5)


# 6. 평가예측 
a = model.evaluate([X_test, x_test], y_test, batch_size=5) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산
print(a)

# model.summary()


# RMSE 구하기
from sklearn.metrics import mean_squared_error

y_predict = model.predict([X_test, x_test], batch_size=5)

def RMSE(y_test, y_predict): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))


# 9. 예측값 구하기(2월 3일)
X_prd = x_prd.reshape(1, 15)
x_prd = x_prd.reshape(1, 15)
X_prd = scaler.transform(x_prd)
x_prd = scaler.transform(x_prd)

X_prd = x_prd.reshape(1, 3, 5)
x_prd = x_prd.reshape(1, 3, 5)

result = model.predict([X_prd, x_prd], batch_size=5)

print(result)

