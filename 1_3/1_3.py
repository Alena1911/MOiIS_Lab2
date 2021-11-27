import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# выводим все столбцы без усечения
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

data_set = pd.read_csv('housing.csv')
print("############ Датасет ############\n",
      data_set, '\r\n')

# 1. Разбейте датасет на тренировочную, валидационную и тестовую выборку
train_val_dt, test_dt = train_test_split(data_set, test_size=0.1)
train_dt, validate_dt = train_test_split(train_val_dt, test_size=0.1)
print("############ Тестовая выборка ############\n",
      test_dt, '\r\n')
print(" ############ Тренировочная выборка ############\n",
      train_dt, '\r\n')
print("############ Валидационная выборка ############\n",
      validate_dt, '\r\n')

# 2. Проведите преобразование категориального признака ocean_proximity через OneHot или Dummy-кодировку
dummy_dt = pd.get_dummies(data_set, columns=['ocean_proximity'])
print("############ Dummy-кодировка ############\n",
      dummy_dt, '\r\n')

# 3. Замените признаки total_rooms и total_bedrooms на average_rooms и average_bedrooms (поделив на households).
data_set['total_rooms'] /= data_set['households']
data_set['total_bedrooms'] /= data_set['households']
data_set.rename(columns={'total_rooms': 'average_rooms', 'total_bedrooms': 'average_bedrooms'}, inplace=True)
print("############ Замена total на average ############\n",
      data_set, '\r\n')

# 4. В признаке average_bedrooms (total_bedrooms) есть отсутствующие значения.
# Определите число экземпляров данных, для которых этот признак отсутствует.
# Придумайте и обоснуйте стратегию заполнения пропусков в этой задаче. Заполните пропуски.
# Заполняем средним количеством комнат. Таким образом количество комнат будет близко к вероятному
print("############ До заполнения ############\n",
      data_set.isna().sum(), '\r\n')  # пропущенные значения
data_set['average_bedrooms'].fillna(value=data_set['average_bedrooms'].mean(), inplace=True)
print("############ После заполнения ############\n",
      data_set.isna().sum(), '\r\n')  # пропущенные значения

# 5. Нормализуйте признаки longitude и latitude
# (сделайте так, чтобы каждый признак имел среднее значение 0 и дисперсию 1 внутри обучающей выборки)
S_S = StandardScaler()
data_set.iloc[:, 0:2] = S_S.fit_transform(data_set.iloc[:, 0:2].to_numpy())
print("############ Нормализовали признаки longitude и latitude ############\n",
      data_set, '\r\n')
