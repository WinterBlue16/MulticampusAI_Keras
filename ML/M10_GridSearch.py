# GridSearch(참고 : https://m.blog.naver.com/PostView.nhn?blogId=gustn3964&logNo=221431933811&categoryNo=13&proxyReferer=https%3A%2F%2Fwww.google.com%2F)
# 제공한 파라미터 중 가장 높은 정확도를 내는 최적의 조건을 찾는다!

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
# scikit-learn 0.20.3에서 31개
# scikit-learn 0.21.2에서 40개 중 4개만 돎 => # 라이브러리 버전 확인 code: print('라이브러리 이름 버전은' + 라이브러리 이름.__version__)

warnings.filterwarnings('ignore')

# 1. 데이터 읽어들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

print(iris_data.head())

# 2. Input, Output 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# 3. train, test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7)

# 4. gridsearch에서 사용할 매개 변수(파라미터) ---(*1)
parameters = [
    {"C": [1, 10, 100, 1000], "kernel": ['linear']},
    {"C": [1, 10, 100, 1000], "kernel": ['rbf'], "gamma": [0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel": ['sigmoid'],"gamma": [0.001, 0.0001]}
]

# 5. gridsearch -- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(SVC(), parameters, cv=kfold_cv)
model.fit(x_train, y_train)
print("최적의 매개 변수 = ", model.best_estimator_) # kfold로 돌린 값 중 가장 좋은 값을 낸 파라미터

# 6. 최적의 매개 변수로 평가하기 ---(*3)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
