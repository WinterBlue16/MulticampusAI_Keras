# 0. 라이브러리 불러오기
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 데이터
x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_train = np.array([0, 1, 1, 0]) 

# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)

# 2. 모델 만들기
from keras.models import Sequential
from keras.layers import Dense
# model = KNeighborsClassifier(n_neighbors=1) # LinearSVC일 땐 안 풀림;

model = Sequential()
model.add(Dense(32, input_shape=(x_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=1)

# 4. 평가 예측
x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_predict = model.predict(x_test)

print(y_predict) # predict값을 1, 0만으로 나올 수 있게 하는 방법?
print("acc=", (model.evaluate(x_test, y_predict)[0])) # accuracy_score(a, b) => a와 b를 비교하여 acc를 측정 # round() 사용 가능
