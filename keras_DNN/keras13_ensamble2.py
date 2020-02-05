# Multi Layer Perceptron
# Ensamble Model 만들기 2(분기 만들기, 다대다 model)


#1. 데이터, 라이브러리 불러오기
import numpy as np

x1 = np.array([range(1, 101), range(101,201), range(301, 401)]) # (3, 100)
x2 = np.array([range(1001, 1101), range(1101,1201), range(1301, 1401)]) 

# y = np.array([range(1, 101)]) # (1, 100) # 대괄호가 있을 경우 벡터가 아닌 행렬이 된다!
# y2 = np.array([range(1001, 1101)]) # input과 output의 갯수는 다를 수 있어도 행은 동일해야 한다!

y1 = np.array([range(1, 101), range(101,201), range(301, 401)]) # (3, 100)
y2 = np.array([range(1001, 1101), range(1101,1201), range(1301, 1401)]) 
y3 = np.array([range(1, 101), range(101,201), range(301, 401)]) 


# 행, 열 변환(reshape, np.transpose, T)
# 참고 : https://rfriend.tistory.com/289
x1 = x1.T # reshape(x, [10,2]) # x = np.transpose(x)
x2 = x2.T 

y1 = y1.T # reshape(y, [10,2]) # y = np.transpose(y)
y2 = y2.T 
y3 = y3.T 


# 2. test, val, train 데이터 나누기
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.6, shuffle=False) # 전체 데이터에서 train, test 분리(꼭 x, y 두 개만 들어갈 필요는 없음!!)
# 입력값은 x(x_train과 x_test로 분리), y(y_train과 y_test로 분리)
x1_val, x1_test, x2_val, x2_test= train_test_split(x1_test, x2_test, test_size=0.5, shuffle=False) # test 데이터의 절반을 validation로 분리

y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, train_size=0.6, shuffle=False) 
y1_val, y1_test, y2_val, y2_test, y3_val, y3_test  = train_test_split(y1_test, y2_test, y3_test, test_size=0.5, shuffle=False)

'''
print(y3_train.shape)
print(y3_test.shape)
print(y3_val.shape)
'''

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

# Concatenate
from keras.layers.merge import concatenate # model을 사슬처럼 엮다.
merge1 = concatenate([output1, output2]) # list 형식(=[]) # merge layer 역시 hidden layer! 따라서 끝 노드 수를 굳이 맞춰주지 않아도 된다.


# Model 3
middle1 = Dense(3)(merge1)
middle2 = Dense(256)(middle1)
middle3 = Dense(1)(middle2) # merge된 마지막 layer

# 분기하기
# 1번째 output model 
output_1 = Dense(64)(middle3)
output_1 = Dense(32)(output_1)
output_1 = Dense(3)(output_1)

# 2번째 output model 
output_2 = Dense(128)(middle3)
output_2 = Dense(256)(output_2)
output_2 = Dense(3)(output_2)

# 3번째 output model
output_3 = Dense(8)(middle3)
output_3 = Dense(4)(output_3)
output_3 = Dense(3)(output_3)


# 함수형 model 선언
model = Model(inputs = [input1, input2],
              outputs = [output_1, output_2, output_3]) # input, output이 다수일 경우 list 형식으로 삽입!

# # model.add(Dense(512, input_dim=3)) # 레이어 추가
# model.add(Dense(256, input_shape=(3,))) # input_shape = (1,), 벡터가 1개라는 뜻
# model.add(Dense(256)) # Node 값 조절 
# model.add(Dense(1)) # regression(=회귀분석) 문제이므로 output은 하나여야 한다. # 열을 기준으로 볼 것!

model.summary() # 모델 구조 확인


#3. 모델 훈련(지표 두 개를 한번에 확인할 수 있다)
from keras.callbacks import EarlyStopping, TensorBoard

td_hist = TensorBoard(log_dir='./graph',
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

model.compile(loss='mse', optimizer='adam', 
              metrics=['acc']) # adam=평타는 침. # metrics는 보여주는 것.
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1, callbacks=[td_hist], validation_data=([x1_val, x2_val], [y1_val, y2_val, y3_val]))


#4. 평가예측 
# a, b, c, d, e, f, g = model.evaluate([x1_test, x2_test],
#                            [y1_test, y2_test, y3_test], batch_size=1) # loss는 자동적으로 출력 # 변수를 7개 주면 ok!
# print(a, b, c, d, e, f, g)

gg = model.evaluate([x1_test, x2_test],
                           [y1_test, y2_test, y3_test], batch_size=1) # loss는 자동적으로 출력
print('gg :', gg)

# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.


# 새로운 데이터 넣어보기
x1_prd = np.array([[501, 502, 503], [601, 602, 603], [701, 702, 703]]).T
x2_prd = np.array([[657, 325, 894], [987, 899, 765], [473, 569, 907]]).T
ggg = model.predict([x1_prd, x2_prd], batch_size=1)
print(ggg)


# RMSE, R2 확인을 위한 변수 생성
y_predict = model.predict([x1_test, x2_test], batch_size=1)


# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(x, y): # 실제 정답값(=y_test), 모델을 통한 예측값(=y_predict) 
    return np.sqrt(mean_squared_error(x, y))

aaa = [RMSE(y1_test, y_predict[0]), RMSE(y2_test, y_predict[1]), RMSE(y3_test, y_predict[2])]
# print(aaa)
RMSE_AVG = np.mean(aaa)
print('RMSE :', RMSE_AVG)

# # 선생님 풀이
# a1 = RMSE(y1_test, y_predict[0])
# a2 = RMSE(y2_test, y_predict[1])
# a3 = RMSE(y3_test, y_predict[2])

# rmse = (a1 + a2 + a3) / 3
# print('RMSE :', rmse)

# R2 구하기
# R2 정의 및 참고 : https://newsight.tistory.com/259
from sklearn.metrics import r2_score

bbb = [r2_score(y1_test, y_predict[0]), r2_score(y2_test, y_predict[1]), r2_score(y3_test, y_predict[2])]
r2_y_predict = np.mean(bbb)

print("R2 :", r2_y_predict)

# # 선생님 풀이
# r1 = r2_score(y1_test, y_predict[0])
# r2 = r2_score(y2_test, y_predict[1])
# r3 = r2_score(y3_test, y_predict[2])

# r2_avg = (r1 + r2 + r3) / 3
# print('R2 score :', rmse)

# visual studio code 참고(https://demun.github.io/vscode-tutorial/shortcuts/)