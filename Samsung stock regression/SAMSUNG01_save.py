# 0. 라이브러리 불러오기
import numpy as np
import pandas as pd

# 1. 데이터 불러오기
df1 = pd.read_csv('./SAMSUNG/samsung.csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
print(df1)
print(df1.shape)

df2 = pd.read_csv('./SAMSUNG/kospi200.csv', index_col=0, header=0, encoding='cp949', sep=',', thousands=',')
print(df2)
print(df2.shape)

df1.info()
df2.info()

# 2. 데이터 전처리
# kospi200의 거래량 
# for i in range(len(df2.index)):
#     df2.iloc[i,4] = int(df2.iloc[i,4].replace(',', '')) # 거래량 str => int 변경

# 삼성전자의 모든 데이터       
# for i in range(len(df1.index)):
#     for j in range(len(df1.iloc[i])):
#         df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))
  
df1 = df1.sort_values(['일자'], ascending=True) # 일자 column 기준으로 오름차순 정렬
df2 = df2.sort_values(['일자'], ascending=True)

df1 = df1.values # np.array로 변경
df2 = df2.values 

print(type(df1), type(df2)) # shape 확인
print(df1.shape, df2.shape)

np.save('./SAMSUNG/DATA/Samsung.npy', arr=df1) # np.array 파일로 저장(원래 데이터로 작업하지 말 것!!!)
np.save('./SAMSUNG/DATA/kospi200.npy', arr=df2)

