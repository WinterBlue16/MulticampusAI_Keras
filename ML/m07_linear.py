from sklearn.datasets import load_boston

# 1. 데이터 불러오기
boston = load_boston()

x = boston.data
y = boston.target

# 2. 데이터 나눠주기
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3. 모델 적용하기
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 3-1 LinearRegression 
model1 = LinearRegression()
model1.fit(x_train, y_train)

# 3-2 Ridge
model2 = Ridge()
model2.fit(x_train, y_train)

# 3-3 Lasso
model3 = Lasso()
model3.fit(x_train, y_train)

# 4. 평가 예측
eee = model1.score(x_test, y_test) # keras의 evaluate에 해당 # 현재는 predict 값과 동일!
ddd = model2.score(x_test, y_test)
sss = model3.score(x_test, y_test) 

print("LinearRegression : %.2f"% eee)
print("Ridge : %.2f"% ddd)
print("Lasso : %.2f"% sss)


