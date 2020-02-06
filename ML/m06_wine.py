# RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 읽어들이기
wine = pd.read_csv("./data/winequality-white.csv", 
                   sep=";", encoding='utf-8')

print(wine.shape)

# 2. Input 데이터, Output 데이터로 분리
y = wine['quality']
x = wine.drop('quality', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# 3. 모델 구성
model = RandomForestClassifier()


# 4. 모델 훈련 
model.fit(x_train, y_train)


# 5. 평가 예측
eee = model.score(x_test, y_test) # keras의 evaluate에 해당 # 현재는 predict 값과 동일!
print("eee : %.2f"% eee)

y_predict = model.predict(x_test)
# print(y_predict)
print("정답률 : %.2f" % accuracy_score(y_predict, y_test)) 


# 6. classification report
print(classification_report(y_test, y_predict))

