
# DNN Modeling 

import numpy as np
from numpy import array

                        # 함수는 keras23_multiple2에서 그대로 copy
def split_sequence3(sequence, n_steps):
    X,y = list(), list()
    for i in range(len(sequence)): 
        end_ix = i + n_steps       
        if end_ix > len(sequence)-1: # 수정
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :] # 수정
        X.append(seq_x) # X 리스트에 추가 
        y.append(seq_y) # y 리스트에 추가 
    return array(X), array(y) # numpy 배열로 만들기

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))]) # 인자를 두 개씩 더한다!

# print(in_seq1.shape)# (10,) 벡터
# print(out_seq.shape)# (10,) 벡터

# 차원 변경(벡터(스칼라 수, ) => 행렬(행, 열))
in_seq1 = in_seq1.reshape(len(in_seq1), 1) # 그냥 10으로 써도 되지만 가능한 변수를 사용하자
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

# print(in_seq1.shape) # (10, 1)
# print(in_seq2.shape) # (10, 1)
# print(out_seq.shape) # (10, 1)

dataset = np.hstack((in_seq1, in_seq2, out_seq)) # 참고 : https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221205882962&proxyReferer=https%3A%2F%2Fwww.google.com%2F
n_steps = 3 

# print(dataset)

x, y = split_sequence3(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], -1) # 가능한 변수를 이용해서 작업할 것! # -1 => 컴퓨터에서 숫자에 맞게 알아서 넣으라는 명령

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)


# 4. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization

input1 = Input(shape=(9,))
dense1 = Dense(256)(input1)
# dense1 = BatchNormalization()
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
# dense1 = Dropout(0.2))
output1 = Dense(3)(dense1)

model = Model(inputs=input1, outputs=output1)

# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)


# 5. 모델 훈련
# from keras.callbacks import EarlyStopping, TensorBoard

# early_stopping = EarlyStopping(monitor='loss', patience=80, mode='auto')
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x_train, y_train, epochs=150, batch_size=1, verbose=2)


# 6. 평가예측 
loss, mse = model.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('loss :', loss)

# model.summary()


# 7. 새로운 데이터 넣어보기
x_prd = array([[90, 95, 105],[100, 105, 115],[110, 115, 125]])
x_prd = x_prd.reshape(1, -1) # -1 => 알아서 넣어주기
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

