#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CNN을 이용한 MNIST 분류
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping 
import numpy as np

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)
# print(y_train)
print(x_train.shape)
print(y_train.shape)

# 2. 데이터 전처리
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# print(type(x_train))

# 2-1. One-hot-Encoding
# softmax를 적용하기 위한 필수적인 과정
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) # 분류 클래스 개수만큼 생성

# 3. model 만들기
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(28, 28)))# activation은 add()가 아닌 Dense()에 포함되어 있다!
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))# activation은 add()가 아닌 Dense()에 포함되어 있다!

model.summary()

# 4. Model 훈련하기
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=20)
model.fit(x_train, y_train, validation_split=0.2, 
          epochs=100, batch_size=8, verbose=1,
          callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)
print(acc)


# In[ ]:




