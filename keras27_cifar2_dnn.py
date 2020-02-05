from keras.datasets import cifar10
from keras.models import Sequential 
from keras.layers import Dense, Flatten, LSTM, Conv2D, MaxPooling2D, Reshape
from keras.callbacks import EarlyStopping 
import numpy as np

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train)
# print(y_train)
print(x_train.shape)
print(y_train.shape)

# 2. 데이터 전처리
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')/255

# print(type(x_train))

# 2-1. One-hot-Encoding
# softmax를 적용하기 위한 필수적인 과정
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) # 분류 클래스 개수만큼 생성

# 3. model 만들기
model = Sequential()
model.add(Dense(512, input_shape=(x_train.shape[1],))) # 가능하면 변수로 바꾸자:)
model.add(Dense(256, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(10, activation='softmax'))# activation은 add()가 아닌 Dense()에 포함되어 있다!

model.summary()

# 4. Model 훈련하기
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=20)
hist = model.fit(x_train, y_train, validation_split=0.2, 
          epochs=100, batch_size=30,
          callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)
print(acc)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('cifar10 DNN loss, acc')
plt.xlabel('epoch')
plt.ylabel('loss, acc')
plt.legend(['train loss', 'test loss', 'train loss', 'test acc'])
plt.show()

# [1.5487799673080445, 0.4557]