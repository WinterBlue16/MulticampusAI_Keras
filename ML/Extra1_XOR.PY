# ML 
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x_train = [[0,0], [1,0], [0,1], [1,1]]
y_train = [0, 1, 1, 0]

model = KNeighborsClassifier(n_neighbors=1)

model.fit(x_train, y_train)

x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)
print("acc = ", accuracy_score([0,1,1,0], y_predict))

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np

x_train = np.array([[0,0], [1,0], [0,1], [1,1]])

from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)


# DNN
model = Sequential()
model.add(Dense(32, input_shape=(x_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[early_stopping])

x_test = np.array([[0,0], [1,0], [0,1], [1,1]])

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)

y_predict = np.where(y_predict > 0.5, 1, 0).reshape(4,) # 핵심!

print(x_test, "의 예측결과 : ", y_predict)
print("acc = ", accuracy_score([0,1,1,0], y_predict))