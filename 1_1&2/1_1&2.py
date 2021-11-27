import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# выводим все столбцы без усечения
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

ds_orders = pd.read_csv('orders.csv')
ds_products = pd.read_csv('products.csv')

data_set = pd.merge(ds_orders, ds_products, how='inner', on='ProductID')
# 1. Определение числа уникальных продуктов в каждой категории
print(data_set.groupby(by='CategoryName')['ProductID'].count().to_string(), '\r\n')
# 2. Вывод всех продуктов категории 'Морепродукты'
print('Продукты категории "Морепродукты":')
print(data_set[data_set['CategoryName'] == 'Морепродукты']['ProductName']
      .drop_duplicates().to_string(index=False), '\r\n')

# 3. График числа заказов за каждый прошедший месяц
copy_df = data_set.copy()
copy_df.index = pd.to_datetime(data_set['OrderDate'])
copy_df.groupby(by=[copy_df.index.month])['Quantity'].count().plot()
plt.title('Число заказов за месяц')
plt.ylabel('Кол-во заказов')
plt.xlabel('Месяц')
plt.show()
# 4.1. Добавление столбца OrderSum
data_set['OrderSum'] = np.where((data_set['UnitPrice_x'] < data_set['UnitPrice_y']),
                                data_set['Quantity'] * data_set['QuantityPerUnit'] * data_set['UnitPrice_x']
                                * (1 - data_set['Discount']),
                                data_set['Quantity'] * data_set['QuantityPerUnit'] * data_set['UnitPrice_y']
                                * (1 - data_set['Discount']))
print('Добавлен OrderSum:')
print(data_set, '\r\n')
# 4.2. Определение самых дорогих заказов
data_set.groupby(by='OrderID')
data_set.sort_values(by='OrderSum', ascending=False, inplace=True)
print('Самые дорогие заказы:')
print(data_set.head(), '\r\n')
# 5. Определение продуктов с максимальной стоимость за шт.
print(data_set.groupby(by='ProductName')['UnitCost'].max().head(), '\r\n')

################ 2-ая задача #######################
# 1. Средний доход от продаж для категорий продуктов
print('Самые доход от продаж для категорий продуктов:')
print(data_set.groupby(by='CategoryName')['OrderSum'].mean(), '\r\n')
# 2. Добавление столбца Profit
data_set['Profit'] = data_set['OrderSum'] - data_set['Quantity'] * data_set['QuantityPerUnit'] * data_set['UnitCost']
print('Добавлен Profit:')
print(data_set, '\r\n')

profit_mean = data_set.groupby(by='CategoryName')['Profit'].sum() / data_set['Profit'].sum()
profit_mean.sort_values(ascending=False, inplace=True)

# Нахождение категорий товаров, обеспечивающих 80% прибыли
print('Категории продуктов, состовляющие 80% прибыли: ')
sum_pr = 0
for i in range(len(profit_mean)):
    sum_pr += profit_mean[i]
    if sum_pr <= 0.8:
        print(f'{profit_mean.index[i]} - {str(profit_mean[i])}')
