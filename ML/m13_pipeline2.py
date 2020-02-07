# Pipeline + Gridsearch
# RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators


warnings.filterwarnings('ignore')

# 1. 데이터 읽어들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

print(iris_data.head())

# 2. Input, Output 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# 3. train, test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.6)

# 4. gridsearch에서 사용할 매개 변수 ---(*1)
parameters = [
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_features":['auto'], "max_depth": [1, 2], "min_samples_split":[2, 3]},
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_features":['sqrt'], "max_depth": [1, 2], "min_samples_split":[2, 3]},
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_features":['log2'], "max_depth": [1, 2], "min_samples_split":[2, 3], "n_jobs": [1, -1]},
]

# 5. gridsearch + Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

kfold_cv = KFold(n_splits=5, shuffle=True)
model = Pipeline([("scaler", MinMaxScaler()), ('RF', RandomForestClassifier())]) # 전처리를 MinMaxScaler로 하고, 그 이름을 'scaler'로 정의
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv)
model.fit(x_train, y_train) # 변수명은 pipe가 아니라 model로 해도 됨!

print("최적의 매개 변수 = ", model.best_estimator_) # pipeline과 gridsearch 위치는 상관없지만 최적 파라미터를 확인하려면 gridsearch가 아래에 오는 게 좋음

# 6. 평가 예측 ---(*3)
y_pred = model.predict(x_test)
print("GridSearch + Pipeline 정답률 = ", accuracy_score(y_test, y_pred))

# 7. 그냥 적용했을 때와 비교(RF)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.score(x_test, y_test)
print("RandomForestClassifier 정확도 : %.2f" % y_predict)

