
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1. 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8', 
                        names=['a', 'b', 'c', 'd', 'y'], header=0) #, header=None

# print(iris_data.shape)

# 2. input 데이터와 output 데이터로 분리하기
from keras.utils import np_utils

y = iris_data["y"]
x = iris_data[['a', 'b', 'c', 'd']]


# 3. 데이터 전처리 
y = y.replace('Iris-setosa', 1)
y = y.replace('Iris-versicolor', 2)
y = y.replace('Iris-virginica', 3)

y = y.values
x = x.values

# print(x.shape)
# print(y.shape)

# 4. train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(x_train.shape)

# 5. 모델 구성

model = Sequential()
model.add(Dense(32, input_shape=(x_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# 6. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1)


# 4. 평가하기 
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy : %.3f"% acc)



