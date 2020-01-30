# 데이터 전처리(Data Scaling, LSTM Model) 

# 0. 라이브러리 불러오기
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 불러오기
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], 
           [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20000, 30000, 40000], [30000, 40000, 50000], [40000, 50000, 60000], [100, 200, 300]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50000, 60000, 70000, 400])

'''
print(x.shape)
print(y.shape)
'''

# 2. 데이터 train, test로 나누기
x_train = x[:10]
x_test = x[10:]
y_train = y[:10]
y_test = y[10:]


# 3. 전처리(preprocessing)를 위한 라이브러리 불러오기
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 
# x를 정규화하면 y는 정규화할 필요가 없다. 
# x, y의 패턴은 일정하게 적용되므로(쌍은 바뀌지 않음) + 목적인 y값이 바뀌면 잘못된 predict값이 도출된다!!!


# 4. 데이터 reshape(reshape를 먼저 해주면 scaler에서 오류가 난다!)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

'''
print(x_train.shape)
print(y_train.shape)
'''

# 5. model 만들기
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

input1 = Input(shape=(3,1))
LSTM1 = LSTM(32, activation='relu', return_sequences=True)(input1) 
LSTM2 = LSTM(256, activation='relu')(LSTM1) 
dense1 = Dense(256)(LSTM2) 
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs = input1, outputs = output1)

# model.summary() 

# 6. model 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae']) 
model.fit(x_train, y_train, epochs=100, batch_size=1)


# 7. 평가예측 
loss, mae = model.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
print('loss :', loss) # 데이터를 순서대로 자른 것이므로 스케일의 차이로 loss값이 폭등한다.


# 8. y_prd 구해보기
x_prd = array([[250, 260, 270]])
scaler.transform(x_prd) 
x_prd = x_prd.reshape(x_prd.shape[0], x_prd.shape[1], 1)
g = model.predict(x_prd, batch_size=1)
print(g)


# 9. R2 구하기
from sklearn.metrics import r2_score

y_predict = model.predict(x_test, batch_size=1)
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)


