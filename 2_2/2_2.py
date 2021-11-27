from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_set = pd.read_excel('Folds5x2_pp.xlsx', sheet_name='Sheet1')

train_val_dt, test_dt = train_test_split(data_set, test_size=0.1)
train_dt, validate_dt = train_test_split(train_val_dt, test_size=0.1)

trainX = train_dt.loc[:, 'AT':'RH'].to_numpy()
trainY = train_dt['PE'].to_numpy()
validateX = validate_dt.loc[:, 'AT':'RH'].to_numpy()
validateY = validate_dt['PE'].to_numpy()
testX = test_dt.loc[:, 'AT':'RH'].to_numpy()
testY = test_dt['PE'].to_numpy()

linear = LinearRegression().fit(trainX, trainY)
print(f'Коэффициент: {linear.score(trainX, trainY)}')

prd = linear.predict(validateX)
print(f'Среднюю ошибку предсказания: {mean_squared_error(prd, validateY)}')

k, b = np.polyfit(validateY, prd, 1)

plt.scatter(validateY, prd)
plt.plot(validateY, k * validateY + b, color='black')
plt.xlabel('Оценка')
plt.ylabel('Наблюдение')
plt.show()