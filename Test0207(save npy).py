# 0. 라이브러리 불러오기
import numpy as np
import pandas as pd

# 1. 데이터 불러오기
df1 = pd.read_csv('./test_data/samsung_data.csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
df2 = pd.read_csv('./test_data/KOSPI.csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
print(df1.shape) # (2458,6)
# print(df2.shape) # (2471,6) # 갯수가 맞지 않음


# 2. 데이터 크기 조정
df2 = df2.sort_values(['Date'], ascending=False) # 데이터 내림차순으로 변경
df2 = df2[:2458] # 오래된 데이터 분리

df2 = df2.sort_values(['Date'], ascending=True) # 데이터 오름차순으로 변경
print(df2.head()) # 데이터 확인
print(df2.tail()) # 최근 데이터인지 확인

# df1.info() # null 없음, float64
# df2.info() # null 없음, float64

# 3. 데이터 전처리
df1 = df1.values # np.array로 변경
# df2 = df2.values 

# print(type(df1), type(df2)) # shape 확인
print(df1)
print(df2)
'''
np.save('./test_data/samsung_data.npy', arr=df1) # np.array 파일로 저장(원래 데이터로 작업하지 말 것!!!)
np.save('./test_data/KOSPI.npy', arr=df2)
'''

# 4. 추가 데이터
df3 = pd.read_csv('./test_data/exchange_rate(CNY).csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
df4 = pd.read_csv('./test_data/exchange_rate(HKD).csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
df5 = pd.read_csv('./test_data/exchange_rate(JPY).csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
print(df3.shape) # (2458,6)
print(df4.shape) # (2458,6)
print(df5.shape) # (2458,6)
