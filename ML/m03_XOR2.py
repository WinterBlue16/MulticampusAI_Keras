# 0. 라이브러리 불러오기
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1. 데이터
x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_train = [0, 1, 1, 0] 


# 2. 모델 만들기
model = KNeighborsClassifier(n_neighbors=1) # LinearSVC일 땐 안 풀림;


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)


print(x_test, "의 예측결과", y_predict)
print("acc=", accuracy_score([0, 1, 1, 0], y_predict)) # accuracy_score(a, b) => a와 b를 비교하여 acc를 측정