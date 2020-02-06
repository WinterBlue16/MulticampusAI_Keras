import pandas as pd
from sklearn.model_selection import train_test_split
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


# 4. classifier 알고리즘 전부 추출하기------(*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier') # 당연히 'regresser'도 존재

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기-------------(*2)
    clf = algorithm()
    
    # 학습하고 평가하기---------------------(*3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name, "의 정답률 = ", accuracy_score(y_test, y_pred))


