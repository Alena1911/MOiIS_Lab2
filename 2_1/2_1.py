import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

data_set = pd.read_csv('Davis.csv')

# удаление колонки индексов, так как она нам не нужна
data_set.drop('Unnamed: 0', axis=1, inplace=True)

# 1. Удаление некорректных данных
data_set = data_set[np.logical_and(data_set['height'] > 100, data_set['weight'] < 150)]

# 2. Выделение тестовой выборки из 50 экземпляров
train_dt, test_dt = train_test_split(data_set, test_size=50)

# 3. Построение гистограмм на тренировочных данных

fig = plt.figure(figsize=(15, 10), dpi=80)

ax_1 = fig.add_subplot(2, 2, 1)
ax_2 = fig.add_subplot(2, 2, 2)
ax_3 = fig.add_subplot(2, 2, 3)
ax_4 = fig.add_subplot(2, 2, 4)

ax_1.hist(train_dt['height'], color='blue', label='Height')
ax_1.hist(train_dt['weight'], color='green', label='Weight')
ax_1.set_title('Weight and height')
ax_1.legend()

ax_2.hist(train_dt[train_dt['sex'] == 'M']['height'], color='blue', label='Male Height')
ax_2.hist(train_dt[train_dt['sex'] == 'M']['weight'], color='brown', label='Male Weight')
ax_2.hist(train_dt[train_dt['sex'] == 'F']['height'], color='brown', label='Female Height', alpha=0.7)
ax_2.hist(train_dt[train_dt['sex'] == 'F']['weight'], color='red', label='Female Weight', alpha=0.7)
ax_2.set_title('Weight and height for sex')
ax_2.legend()

# Меняем в выбрках М на 0, а F на 1
train_dt.replace({'M': 0, 'F': 1}, inplace=True)
test_dt.replace({'M': 0, 'F': 1}, inplace=True)

trainX = train_dt.loc[:, 'weight':'height'].to_numpy()
trainY = train_dt['sex'].to_numpy()
clf_train = LogisticRegression().fit(trainX, trainY)
print(f'Производительность тренировочной: {clf_train.score(trainX, trainY)}')

testX = test_dt.loc[:, 'weight':'height'].to_numpy()
testY = test_dt['sex'].to_numpy()
clf_test = LogisticRegression().fit(testX, testY)
print(f'Производительность тестовой: {clf_test.score(testX, testY)}')

predicts = clf_train.predict(trainX)

ax_3.scatter(trainX[predicts == 0][:, 1], trainX[predicts == 0][:, 0], color='blue', label='M')
ax_3.scatter(trainX[predicts == 1][:, 1], trainX[predicts == 1][:, 0], color='red', label='F')
ax_3.set_ylabel('weight')
ax_3.set_xlabel('height')
ax_3.set_title('Тренировочная выборка')
ax_3.legend()


predicts = clf_test.predict(testX)
ax_4.scatter(testX[predicts == 0][:, 1], testX[predicts == 0][:, 0], color='blue', label='M')
ax_4.scatter(testX[predicts == 1][:, 1], testX[predicts == 1][:, 0], color='red', label='F')
ax_4.set_ylabel('weight')
ax_4.set_xlabel('height')
ax_4.set_title('Тестовая выборка')
ax_4.legend()

# попытка закрасить области
# fig1, ax_test = plt.subplots()
# svm = SVC(C=0.5, kernel='linear')
# svm.fit(trainX, trainY)
# plot_decision_regions(trainX, trainY, clf=svm, legend=2, filler_feature_values={0: 'M', 1: 'F'})

plt.show()
