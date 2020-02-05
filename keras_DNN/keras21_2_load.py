# 모델 불러와 적용하기

#1. 데이터, 라이브러리 불러오기
import numpy as np

x = np.array([range(1, 101), range(101,201), range(301, 401)]) # (3, 100)
y = np.array([range(1, 101)]) # (1, 100) # 대괄호가 있을 경우 벡터가 아닌 행렬이 된다!

x = x.T # reshape(x, [10,2]) # x = np.transpose(x)
y = y.T # reshape(y, [10,2]) # y = np.transpose(y)

# 2. test, val, train 데이터 나누기(모델을 불러오더라도 이거 빼먹으면 안됨!)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False) # 전체 데이터에서 train, test 분리
# 입력값은 x(x_train과 x_test로 분리), y(y_train과 y_test로 분리)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False) # test 데이터의 절반을 validation로 분리


#2. 모델구성
from keras.models import load_model, Model
from keras.layers import LSTM, Dense, Input
model = load_model('./save/savetest01.h5')


# #3. 모델 훈련
# from keras.callbacks import EarlyStopping, TensorBoard

# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)

# early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
# model.compile(loss='mse', optimizer='adam', 
#               metrics=['mae']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# model.summary()

# 3.1. Layer 추가해 모델 수정하기
# 참고(https://rarena.tistory.com/entry/keras-%ED%8A%B9%EC%A0%95-%EB%AA%A8%EB%8D%B8%EB%A1%9C%EB%93%9C%ED%95%98%EC%97%AC-%EB%82%B4-%EB%A0%88%EC%9D%B4%EC%96%B4, https://stackoverflow.com/questions/56456471/valueerror-the-name-dense-1-is-used-2-times-in-the-model-all-layer-names-sho)
# input1 = Input(shape=(1,))
input1 = model.output
dense = Dense(256, name='dense_x')(input1) # Layer 이름을 새로 지정해줘야 한다!! default=dense_1 
dense2 = Dense(256, name='dense_y')(dense)
dense3 = Dense(256, name='dense_z')(dense2)
output1 = Dense(1, name='output_final')(dense3)

model_f = Model(inputs=model.input, outputs=output1)

# model.add(Dense(256))
# model.add(Dense(256))
# model.add(Dense(256))
# model.add(Dense(1))

#3. 모델 훈련
from keras.callbacks import EarlyStopping, TensorBoard

td_hist = TensorBoard(log_dir='./graph',
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model_f.compile(loss='mse', optimizer='adam', 
              metrics=['mae']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model_f.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[td_hist], validation_data=(x_val, y_val))

#4. 평가예측 
loss, mse = model_f.evaluate(x_test, y_test, batch_size=1) # loss는 자동적으로 출력
# mae = mean_absolute_error, mse와 다른 손실함수
# 데이터 크기보다 더 큰 batch size를 줄 경우 데이터 크기로 계산된다.
print('mse :', mse)

model_f.summary()

# 새로운 데이터 넣어보기
x_prd = np.array([[290, 974, 467], [235, 436, 765], [473, 569, 907]]).T
g = model_f.predict(x_prd, batch_size=1)
print(g)

y_predict = model_f.predict(x_test, batch_size=1)


# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 실제 정답값, 모델을 통한 예측값 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))


# R2 구하기
# R2 정의 및 참고 : https://newsight.tistory.com/259
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)


