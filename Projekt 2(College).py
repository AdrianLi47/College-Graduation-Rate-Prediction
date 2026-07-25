# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

dane = pd.read_csv("College.csv")
dane.columns.values[0] = 'College_Names'
dane

# COMMAND ----------

dane = dane.dropna()
dane['Grad.Rate'] = dane['Grad.Rate'].clip(upper=100)
dane['PhD'] = dane['PhD'].clip(upper=100)
dane['Terminal'] = dane['Terminal'].clip(upper=100)
dane.shape

# COMMAND ----------

dane.describe()

# COMMAND ----------

#podzial zmiennych na kat i num
var_num = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend']
var_cat = ['Private']

# COMMAND ----------

korelacja = dane[var_num + ['Grad.Rate']].corr()['Grad.Rate'].drop('Grad.Rate')
korelacja.sort_values(ascending=False)

# COMMAND ----------

plt.figure(figsize=(12, 9))
korelacja_heatmap = dane[var_num + ['Grad.Rate']].corr()

sns.heatmap(korelacja_heatmap, annot=True, fmt='.2f', cmap='Blues', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.6})
plt.title('Korelacja między zmiennymi a odsetkiem kończących studia', fontsize='14',pad='20' )
plt.savefig('korelacja.png')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Outstate najwyższa wartość korelacji w stosunku do odsetku kończących studia

# COMMAND ----------

# MAGIC %md
# MAGIC #Podział danych na zbiór uczący i testowy

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

y = dane['Grad.Rate']
X = dane.drop(['College_Names', 'Grad.Rate'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=327734)

# COMMAND ----------

X_train.shape

# COMMAND ----------

X_test.shape

# COMMAND ----------

#kopia zap danych num
stand_X_train = X_train.copy()
stand_X_test = X_test.copy()

# COMMAND ----------

#skalowanie danych num
from sklearn.preprocessing import StandardScaler

skaler = StandardScaler()
stand_X_train[var_num] = skaler.fit_transform(stand_X_train[var_num]) 
stand_X_test[var_num] = skaler.transform(stand_X_test[var_num])

# COMMAND ----------

#skalowanie danych kat

from sklearn.preprocessing import OneHotEncoder

enkoder = OneHotEncoder(sparse_output = False, drop = 'first').set_output(transform='pandas')

temp = enkoder.fit_transform(stand_X_train[var_cat])
X_train = stand_X_train.drop(columns = var_cat)
X_train = pd.concat([X_train,temp],axis=1)

temp = enkoder.transform(stand_X_test[var_cat])
X_test = stand_X_test.drop(columns = var_cat)
X_test = pd.concat([X_test,temp],axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #Model Regresji Liniowej

# COMMAND ----------

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# COMMAND ----------

model_reg_lin = LinearRegression()
model_reg_lin.fit(X_train, y_train)

# COMMAND ----------

y_pred_reg_lin=model_reg_lin.predict(X_test)

# COMMAND ----------

pd.DataFrame(model_reg_lin.coef_, index = model_reg_lin.feature_names_in_, columns = ['współczynnik modelu'])

# COMMAND ----------

# MAGIC %md
# MAGIC ####Rownanie modelu 

# COMMAND ----------

print('karp = ', end='')
for b,n in zip(model_reg_lin.coef_, model_reg_lin.feature_names_in_):
    print(round(b,2), '*', n, '+ ',end='') 
print(round(model_reg_lin.intercept_,2))

# COMMAND ----------

ocena_R2 = model_reg_lin.score(X_test, y_test)
mae_reg_lin = mean_absolute_error(y_test, y_pred_reg_lin)
mse_reg_lin = mean_squared_error(y_test, y_pred_reg_lin)
rmse_reg_lin = np.sqrt(mse_reg_lin)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Ocena modelu Regresji Liniowej

# COMMAND ----------

print(f'Dopasowanie modelu regresji liniowej R² wynosi: {round(ocena_R2 * 100, 2)}')
print(f'Średni błąd absolutny MAE wynosi: {round(mae_reg_lin, 2)}')
print(f'Pierwiastek z błędu średniokwadratowego RMSE wynosi: {round(rmse_reg_lin, 2)}')


# COMMAND ----------

# MAGIC %md
# MAGIC #Model Random Forest

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

model_rf = RandomForestRegressor(n_estimators=100, random_state=327734)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

ocena_R2_rf = model_rf.score(X_test, y_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f'Dopasowanie modelu Random Forest R² wynosi: {round(ocena_R2_rf*100, 2)}%')
print(f'Średni błąd absolutny MAE wynosi: {round(mae_rf, 2)}')
print(f'Pierwiastek z błędu średniokwadratowego RMSE wynosi: {round(rmse_rf, 2)}')

# COMMAND ----------

model_rf200 = RandomForestRegressor(n_estimators=200, random_state=327734)
model_rf200.fit(X_train, y_train)

y_pred_rf200 = model_rf200.predict(X_test)

ocena_R2_rf200 = model_rf200.score(X_test, y_test)
mae_rf200 = mean_absolute_error(y_test, y_pred_rf)
mse_rf200 = mean_squared_error(y_test, y_pred_rf)
rmse_rf200 = np.sqrt(mse_rf)

print(f'Dopasowanie modelu Random Forest R² wynosi: {round(ocena_R2_rf200*100, 2)}%')
print(f'Średni błąd absolutny MAE wynosi: {round(mae_rf200, 2)}')
print(f'Pierwiastek z błędu średniokwadratowego RMSE wynosi: {round(rmse_rf200, 2)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Tune Modelu za pomocą GridSearchCV

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf_base = RandomForestRegressor(random_state=327734)


grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

print(f'Testowanie {len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_split"]) * len(param_grid["max_features"])} kombinacji parametrów')

grid_search.fit(X_train, y_train)

print('Najlepsze Parametry:\n')
for param, value in grid_search.best_params_.items():
    print(f'{param}: {value}')

# COMMAND ----------

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

ocena_R2_best = best_model.score(X_test, y_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

# COMMAND ----------

print(f'Dopasowanie modelu Random Forest dla najlepszych parametrów: {round(ocena_R2_best*100, 2)}%')
print(f'Poprawa o {round((ocena_R2_best - ocena_R2_rf)*100, 2)}pkt %\n')
print(f'Średni błąd absolutny MAE dla ulepszonego modelu Random Forest wynosi:  {round(mae_best, 2)}\n')
print(f'Pierwiastek z błędu średniokwadratowego RMSE dla ulepszonego modelu Random Forest wynosi: {round(rmse_best, 2)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Eksport zbioru testowego i uczącego oraz wyników predykcji modelu regresji liniowej

# COMMAND ----------

train_set = X_train.copy()
train_set['Grad.Rate'] = y_train
train_set=pd.DataFrame(train_set)

test_set = X_test.copy()
test_set['Grad.Rate'] = y_test
test_set=pd.DataFrame(test_set)

train_set.to_csv('train_set.csv')
test_set.to_csv('test_set.csv')

# COMMAND ----------

y_pred_csv = pd.DataFrame(y_pred_reg_lin)
y_pred_csv.to_csv('Wyniki_predykcji_reg_lin.csv')

# COMMAND ----------

