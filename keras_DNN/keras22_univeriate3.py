# dataset을 한번에 분리하는 함수

import numpy as np
from numpy import array
                        # dataset, 4
def split_sequence(sequence, n_steps):
    X,y = list(), list()
    for i in range(len(sequence)): # 10
        end_ix = i + n_steps       # 0 + 4 = 4
        if end_ix > len(sequence)-1: # 4 > 9
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] # [0:4], [4] => 0, 1, 2, 3 / 4
        X.append(seq_x) # X 리스트에 추가 # 0, 1, 2, 3
        y.append(seq_y) # y 리스트에 추가 # 4
    return array(X), array(y) # numpy 배열로 만들기


dataset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_steps = 3

x, y = split_sequence(dataset, n_steps)
'''
print(x)
print(y)
'''

for i in range(len(x)):
    print(x[i], y[i])


# DNN 모델 구성
# loss 출력
# x_prd = array([90, 100, 110]) 

'''
print(x.shape)
print(y.shape)
'''
x = x.reshape(x.shape[0], x.shape[1], 1)

# 3. 데이터 나누기
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)


# 4. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization


model = Sequential()

model.add(LSTM(256, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(LSTM(256, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
# model.add(Dropout(0.2))
model.add(Dense(1))

# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)


# 5. 모델 훈련
# from keras.callbacks import EarlyStopping, TensorBoard

# early_stopping = EarlyStopping(monitor='loss', patience=80, mode='auto')
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x_train, y_train, epochs=100, batch_size=1)


# 6. 평가예측 
loss, mse = model.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('loss :', loss)

# model.summary()


# 7. 새로운 데이터 넣어보기
x_prd = array([[90, 100, 110]])
x_prd = x_prd.reshape(x_prd.shape[0], x_prd.shape[1], 1)
g = model.predict(x_prd, batch_size=1)
print(g)

y_predict = model.predict(x_test, batch_size=1)


# 8. RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))



# 9. R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)


