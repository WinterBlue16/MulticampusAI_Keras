# RandomizedSearch로 변경

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

'''
# 4. Randomizedsearch에서 사용할 매개 변수 ---(*1)
parameters = [
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_features":['auto'], "max_depth": [1, 2], "min_samples_split":[2, 3]},
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_features":['sqrt'], "max_depth": [1, 2], "min_samples_split":[2, 3]},
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_features":['log2'], "max_depth": [1, 2], "min_samples_split":[2, 3], "n_jobs": [1, -1]},
    {"n_estimators": [70, 75, 80, 85, 90, 95, 100], "max_depth": [1, 2, 3],"n_jobs": [1, -1]}
]


# 5. RandomizedSearch -- (*1)
kfold_cv = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions={"n_estimators": [70, 75, 80, 85, 90, 95, 100], 
                                                "max_features":['sqrt'], 
                                                "max_depth": [1, 2], 
                                                "min_samples_split":[2, 3]}, cv=kfold_cv)
model.fit(x_train, y_train)

print("최적의 매개 변수 = ", model.best_estimator_) # kfold로 돌린 값 중 가장 좋은 값을 낸 파라미터

# 6. 최적의 매개 변수로 평가하기 ---(*2)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))

# 최적의 매개 변수 =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='sqrt', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
# 최종 정답률 =  1.0

'''
# 7. 그냥 적용했을 때와 비교(0.9666666666666667)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.score(x_test, y_test)
print("%.2f" % y_predict)
