from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 불러오기
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], 
           [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20000, 30000, 40000], [30000, 40000, 50000], [40000, 50000, 60000], [100, 200, 300]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50000, 60000, 70000, 400])

'''
print(x.shape)
print(y.shape)
'''

# 2. 전처리(preprocessing)를 위한 라이브러리 불러오기
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # 함수 정의 # 모든 데이터값을 최솟값(=0)과 최댓값(=1)사이의 값으로 채워넣는다!
scaler.fit(x)
x = scaler.transform(x)
print(x)

