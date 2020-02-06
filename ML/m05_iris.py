import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8', 
                        names=['a', 'b', 'c', 'd', 'y']) #, header=None

# 2. input 데이터와 output 데이터로 분리하기
y = iris_data.loc[:, "y"]
x = iris_data.loc[:, ['a', 'b', 'c', 'd']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7)

# 3. 모델 학습
clf = SVC()
clf.fit(x_train, y_train)

# 4. 평가하기 
y_pred = clf.predict(x_test)
print("정답률 :", accuracy_score(y_test, y_pred))