import matplotlib.pyplot as plt
import pandas as pd


# 1. 데이터 읽어들이기
wine = pd.read_csv("./data/winequality-white.csv", 
                   sep=";", encoding='utf-8')

# print(wine.head())

# 2. 데이터 살피기
count_data = wine.groupby('quality')['quality'].count()
print(count_data)

count_data.plot()
plt.savefig('wine-count-plt.png')
plt.show()

