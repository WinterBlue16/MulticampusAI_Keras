# 차원축소
# 데이터 전처리 => 차원축소로 모델 한 번 돌려보기 => feature importance

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("원본 데이터 형태 :", X_scaled.shape)
print("축소된 데이터 형태 :", X_pca.shape)
