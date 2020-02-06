# RandomForest
# Deeplearning 구현 시 machinelearning의 정확도를 따라잡는 것이 기본이다!

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import numpy as np

# 1. 데이터 읽어들이기
wine = pd.read_csv("./data/winequality-white.csv", 
                   sep=";", encoding='utf-8', header=0)

# print(wine)


# 2. Input 데이터, Output 데이터로 분리
y = wine['quality']
x = wine.drop('quality', axis=1)

y = y.values
x = x.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train) # one-hot-encoding 수정 => sklearn 사용!
# y_test = np_utils.to_categorical(y_test)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
y_train = y_train.reshape(len(y_train), -1) # reshape 변환이 반드시 필요하다!!
y_test = y_test.reshape(len(y_test), -1)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# print(y_test.shape)
# print(y_train.shape)


# 5. 모델 구성

model = Sequential()
model.add(Dense(256, input_shape=(x_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.summary()

# 6. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)


# 4. 평가하기 
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy : %.3f"% acc)


# Accuracy : 0.514