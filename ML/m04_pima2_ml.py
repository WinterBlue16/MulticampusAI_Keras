# LinearSVC, KNeighborsClassifier 

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# 1. 데이터 불러오기
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]


# 2. 데이터 나눠주기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)


# 3. 모델 설정
model = LinearSVC()


# 4. 모델 학습
model.fit(x_train,y_train)


# 5. 결과 출력 
y_predict = model.predict(x_test)


print(x_test, "의 예측결과", y_predict)
print("Accuracy = %.2f" % accuracy_score(y_test, y_predict))

