# 데이터 전처리(Data Scaling)

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

# 2. 전처리(preprocessing)를 위한 라이브러리 불러오기
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

# 2.1. MinMaxScaler
# scaler = MinMaxScaler() # 함수 정의 # 모든 데이터값을 최솟값(=0)과 최댓값(=1)사이의 값으로 채워넣는다!
# scaler.fit(x) # 훈련
# x = scaler.transform(x) # 적용
# print(x)

# 2.2 StandardScaler
# scaler = StandardScaler() # 데이터값 - 평균 / 표준편차
# scaler.fit(x) 
# x = scaler.transform(x)  
# print(x)

# # 2.3 RobustScaler
scaler = RobustScaler() # 중앙값(median)과 IQR 사용(참고: https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221021173204&proxyReferer=https%3A%2F%2Fwww.google.com%2F) 
# StandradScaler와 동일한 값을 보다 넓게 분포
scaler.fit(x) 
scaler.transform(x) 
# print(x)

# # 2.4 MaxAbsScaler
# scaler = MaxAbsScaler() # 데이터값을 -1~1 사이로 재조정한다 
# scaler.fit(x) 
# x = scaler.transform(x) 
# print(x)

# 2.5 MinMax + Standard(바꿔서 하는 것도 가능!)
# scaler = StandardScaler() 
# scaler.fit(x) 
# x = scaler.transform(x) 

# scaler = MinMaxScaler() 
# scaler.fit(x) 
# x = scaler.transform(x) 

# print(x)

# x를 정규화하면 y는 정규화할 필요가 없다. 
# x, y의 패턴은 일정하게 적용되므로(쌍은 바뀌지 않음) + 목적인 y값이 바뀌면 잘못된 predict값이 도출된다!!!

# 3. Data split
# train 10개, 나머지는 test data
# sklearn의 train_test_split을 쓸 때 반드시 shuffle=True를 적용, 데이터가 섞이도록 해준다. 

x_train = x[:10]
x_test = x[10:]
y_train = y[:10]
y_test = y[10:]

scaler = RobustScaler() 
scaler.fit(x_train) 
scaler.transform(x_train) 
scaler.transform(x_test) 

print(x_train.shape)
print(y_train.shape)

# 4. model 만들기
from keras.models import Sequential 
from keras.layers import Dense  

model=Sequential()

model.add(Dense(32, input_shape=(3,))) 
model.add(Dense(256)) 
model.add(Dense(256)) 
model.add(Dense(256)) 
model.add(Dense(256)) 
model.add(Dense(256)) 
model.add(Dense(256))
model.add(Dense(1))

# model.summary() 

# 5 . model 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae']) 
model.fit(x_train, y_train, epochs=200, batch_size=1)


#4. 평가예측 
loss, mae = model.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
print('loss :', loss) # 데이터를 순서대로 자른 것이므로 스케일의 차이로 loss값이 폭등한다.

x_prd = array([[250, 260, 270]])
scaler.transform(x_prd) 
g = model.predict(x_prd, batch_size=1)
print(g)

# R2 구하기
from sklearn.metrics import r2_score

y_predict = model.predict(x_test, batch_size=1)
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)


