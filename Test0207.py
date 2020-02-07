import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 3. 저장한 np.array 파일 불러오기
samsung = np.load('./test_data/samsung_data.npy')
# kospi = np.load('./test_data/KOSPI.npy')

print(samsung) # (2458,6) => (2451,)
# print(kospi) # (2458,6) => (2451,)

def split_xy5(dataset, time_steps, y_column): 
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

time_steps = 7
x1, y1 =split_xy5(samsung, time_steps, 1)
# x2, y2 =split_xy5(kospi, time_steps, 1) # y2값은 사용하지 않음


# 3. 데이터 split
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.7, random_state=1, shuffle=False)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.7, random_state=1, shuffle=False)

tree = RandomForestClassifier(random_state=0)
tree.fit(x1_train, y1_train)
print("train set 정확도 :{:.3f}".format(tree.score(x1_train, y1_train)))
print("test set 정확도 :{:.3f}".format(tree.score(x1_test, y1_test)))

print("특성 중요도:\n", tree.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = x1.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), x1.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
    
plot_feature_importances_x1(tree)
plt.show()

# 4. 데이터 전처리(3차원 => 2차원)
from sklearn.preprocessing import StandardScaler

# 데이터 2차원으로 변환
x1_train = x1_train.reshape(x1_train.shape[0], -1) 
x1_test = x1_test.reshape(x1_test.shape[0], -1) 
# x2_train = x2_train.reshape(x2_train.shape[0], -1)
# x2_test = x2_test.reshape(x2_test.shape[0], -1) 

scaler = StandardScaler() # 데이터값 - 평균 / 표준편차

scaler.fit(x1_train) 
x1_train_scaled = scaler.transform(x1_train)  
x1_test_scaled = scaler.transform(x1_test)

# scaler.fit(x2_train) 
# x2_train_scaled = scaler.transform(x2_train)  
# x2_test_scaled = scaler.transform(x2_test)
'''
# 데이터 3차원으로 재변경
x1_train_scaled = x1_train_scaled.reshape(x1_train.shape[0], 7, 6) # 다시 3차원 변경(LSTM 사용 시)
x1_test_scaled = x1_test_scaled.reshape(x1_test.shape[0], 7, 6) 
# x2_train_scaled = x2_train_scaled.reshape(x2_train.shape[0], 7, 6) 
# x2_test_scaled = x2_test_scaled.reshape(x2_test.shape[0], 7, 6) 

# print(x1_train_scaled.shape)
# print(x1_test_scaled.shape)
# print(x2_train_scaled.shape)
# print(x2_test_scaled.shape)


# 3. 모델 구성
# LSTM ensamble
from keras.models import Model
from keras.layers import Input, Dropout, BatchNormalization, Dense, LSTM

# model1(samsung data)
input1 = Input(shape=(7, 6))
dense1 = LSTM(256, activation='relu', return_sequences=True)(input1)
dense1 = LSTM(256, activation='relu')(dense1)
dense1 = Dense(256)(dense1)
dense1 = BatchNormalization()(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dropout(0.3)(dense1)
output1 = Dense(1)(dense1)

# model2(kospi data)
# input2 = Input(shape=(7, 6))
# dense2 = LSTM(256, activation='relu', return_sequences=True)(input2)
# dense2 = LSTM(256, activation='relu')(dense2)
# dense2 = Dense(256)(dense2)
# dense2 = BatchNormalization()(dense2)
# dense2 = Dense(256)(dense2)
# dense2 = Dense(256)(dense2)
# dense2 = Dropout(0.3)(dense2)
# output2 = Dense(8)(dense2)

# Concatenate
# from keras.layers.merge import concatenate
# merge1 = concatenate([output1,output2])

# # ensamble model 1
# dense3 = Dense(256)(merge1) 
# dense3 = Dense(256)(dense3) 
# dense3 = Dense(256)(dense3) 
# output3 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
'''

# ML Model
# from sklearn.utils.testing import all_estimators

# allAlgorithms = all_estimators(type_filter='regressor') 

# print(allAlgorithms)
# print(len(allAlgorithms))
# print(type(allAlgorithms))

# for (name, algorithm) in allAlgorithms:

#     model = algorithm()
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     print(name, "의 loss = ", r2_score(y_test, y_pred))
model = RandomForestClassifier()
model.fit(x1_train, y1_train)


'''
# 5. 모델 훈련
from keras.callbacks import EarlyStopping, TensorBoard
# td_hist = TensorBoard(log_dir='./graph',
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)

early_stopping = EarlyStopping(monitor='loss', patience=60, mode='auto')
model.compile(loss='mae', 
              optimizer='adam', 
              metrics=['mse']) # adam=평타는 침. # 이 때문에 아래서 acc가 나온다.
model.fit(x1_train_scaled, y1_train, 
          epochs=10, 
          batch_size=10, 
          validation_split=0.2, 
          callbacks=[early_stopping])

# 6. 평가예측 
loss, mse = model.evaluate(x1_test_scaled, 
                           y1_test, 
                           batch_size=10) # loss는 자동적으로 출력
print('loss :', loss)
print('mse :', mse)
'''
# model.summary()

# 7. RMSE 구하기
from sklearn.metrics import mean_squared_error

y_predict = model.predict(x1_test_scaled)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
print('RMSE :', RMSE(y1_test, y_predict))


# 9. 예측값 구하기
x1_prd = samsung[2451:] # (7, 6)
# x2_prd = kospi[2451:] # (7, 6)

x1_prd = x1_prd.reshape(1, -1)
# x2_prd = x2_prd.reshape(1, -1)
x1_prd_scaled = scaler.transform(x1_prd)
# x2_prd_scaled = scaler.transform(x2_prd)
# x1_prd_scaled = x1_prd_scaled.reshape(1, 7, 6)
# x2_prd_scaled = x2_prd_scaled.reshape(1, 7, 6)

result = model.predict(x1_prd_scaled)
print(result)

# 2월 7일 예상 주가 : -2.7748378e+09
