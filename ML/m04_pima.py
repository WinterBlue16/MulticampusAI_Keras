from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 모델 설정
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'])

# 모델 실행
model.fit(X,Y, epochs=200, batch_size=10)

# 결과 출력 
print("\n Accuracy: %.3f"% (model.evaluate(X, Y)[1]))

