import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# выводим все столбцы без усечения
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


# замена total на average
def total_to_average(df):
    df['total_rooms'] /= df['households']
    df['total_bedrooms'] /= df['households']
    df.rename(columns={'total_rooms': 'average_rooms', 'total_bedrooms': 'average_bedrooms'}, inplace=True)


data_set = pd.read_csv('housing.csv')

# 2. Проведите преобразование категориального признака ocean_proximity через OneHot или Dummy-кодировку
data_set = pd.get_dummies(data_set, columns=['ocean_proximity'])
print("############ Dummy-кодировка ############\n",
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

# 3. Замените признаки total_rooms и total_bedrooms на average_rooms и average_bedrooms (поделив на households).
total_to_average(test_dt)
total_to_average(train_dt)
total_to_average(validate_dt)

# 4. В признаке average_bedrooms (total_bedrooms) есть отсутствующие значения.
# Определите число экземпляров данных, для которых этот признак отсутствует.
# Придумайте и обоснуйте стратегию заполнения пропусков в этой задаче. Заполните пропуски.
# Заполняем средним количеством комнат. Таким образом количество комнат будет близко к вероятному
# print("############ До заполнения ############\n",
#       data_set.isna().sum(), '\r\n')  # пропущенные значения
test_dt['average_bedrooms'].fillna(value=test_dt['average_bedrooms'].mean(), inplace=True)
train_dt['average_bedrooms'].fillna(value=train_dt['average_bedrooms'].mean(), inplace=True)
validate_dt['average_bedrooms'].fillna(value=validate_dt['average_bedrooms'].mean(), inplace=True)
# print("############ После заполнения ############\n",
#       data_set.isna().sum(), '\r\n')  # пропущенные значения

# 5. Нормализуйте признаки longitude и latitude
# (сделайте так, чтобы каждый признак имел среднее значение 0 и дисперсию 1 внутри обучающей выборки)
S_S = StandardScaler()
train_dt.loc[:, 'longitude':'latitude'] = S_S.fit_transform(train_dt.loc[:, 'longitude':'latitude'].to_numpy())
test_dt.loc[:, 'longitude':'latitude'] = S_S.transform(test_dt.loc[:, 'longitude':'latitude'].to_numpy())
validate_dt.loc[:, 'longitude':'latitude'] = S_S.transform(validate_dt.loc[:, 'longitude':'latitude'].to_numpy())

print("############ Нормализовали признаки longitude и latitude ############\n",
      train_dt, test_dt, validate_dt)
