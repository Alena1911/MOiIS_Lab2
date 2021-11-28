import pandas as pd
import matplotlib.pyplot as plt

# выводим все столбцы без усечения
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

ds_orders = pd.read_csv('orders.csv')
ds_products = pd.read_csv('products.csv')

# 1. Определение числа уникальных продуктов в каждой категории
print(ds_products.groupby(by='CategoryName')['ProductID'].count().to_string(), '\r\n')
# 2. Вывод всех продуктов категории 'Морепродукты'
print('Продукты категории "Морепродукты":')
print(ds_products[ds_products['CategoryName'] == 'Морепродукты']['ProductName'].to_string(index=False), '\r\n')

# 3. График числа заказов за каждый прошедший месяц
copy_df = ds_orders.copy()
copy_df['OrderDate'] = pd.to_datetime(ds_orders['OrderDate'])
copy_df.groupby(by=[copy_df['OrderDate'].dt.year, copy_df['OrderDate'].dt.month])['OrderID'].count().plot()
plt.title('Число заказов за месяц')
plt.ylabel('Кол-во заказов')
plt.xlabel('Месяц')
plt.show()

# 4.1. Добавление столбца OrderSum
ds_orders['OrderSum'] = ds_orders['UnitPrice'] * ds_orders['Quantity'] * (1 - ds_orders['Discount'])
print('Добавлен OrderSum:')
print(ds_orders, '\r\n')

# 4.2. Определение самых дорогих заказов
ds_orders.groupby(by='OrderID')
ds_orders.sort_values(by='OrderSum', ascending=False, inplace=True)
print('Самые дорогие заказы:')
print(ds_orders.head(), '\r\n')

# 5. Определение продуктов с максимальной стоимость за шт.
ds_products.groupby(by='ProductName')
ds_products.sort_values(by='UnitCost', ascending=False, inplace=True)
print('Самые дорогие продукты:')
print(ds_products.head(), '\r\n')

################ 2-ая задача #######################
# 1. Средний доход от продаж для категорий продуктов
data_set = pd.merge(ds_orders, ds_products, how='inner', on='ProductID')
data_set['OrderSum'] = data_set['UnitPrice_x'] * data_set['Quantity'] * (1 - data_set['Discount'])
print('Средний доход от продаж для категорий продуктов:')
print(data_set.groupby(by='CategoryName')['OrderSum'].mean().sort_values(ascending=False), '\r\n')
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
