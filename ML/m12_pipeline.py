# Pipeline 

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
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

# 4. 데이터 전처리
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())]) # 전처리를 MinMaxScaler로 하고, 그 이름을 'scaler'로 정의
# 모델은 SVC를 적용하고, 이름은 'SVM'이라고 정의
pipe.fit(x_train, y_train) # 변수명은 pipe가 아니라 model로 해도 됨!

print("Test(pipeline) 점수 : %.3f" % pipe.score(x_test, y_test))

'''
# 4. Randomizedsearch에서 사용할 매개 변수 ---(*1)
parameters = {"n_estimators": [70, 75, 80, 85, 90, 95, 100], 
                                                "max_features":['sqrt'], 
                                                "max_depth": [1, 2], 
                                                "min_samples_split":[2, 3]}

# 5. RandomizedSearch -- (*1)
kfold_cv = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=parameters, cv=kfold_cv)
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
# 7. 그냥 적용했을 때와 비교(SVC)
model = SVC()
model.fit(x_train, y_train)

y_predict = model.score(x_test, y_test)
print("SVC 정확도 : %.2f" % y_predict)

