# DecisionTree 

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("train set 정확도 :{:.3f}".format(tree.score(X_train, y_train)))
print("test set 정확도 :{:.3f}".format(tree.score(X_test, y_test)))

print("특성 중요도:\n", tree.feature_importances_)