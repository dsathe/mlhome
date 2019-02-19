import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Getting the datasets using the pandas library
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
from sklearn.cross_validation import train_test_split
X=df.iloc[:,:-1].values
y=df['MEDV'].values
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=1)
#tree=DecisionTreeRegressor(max_depth=2)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)


from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))


from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


plt.scatter(y_train_pred,
y_train_pred - y_train,
c='black',
marker='o',
s=35,
alpha=0.5,
label='Training data')
plt.scatter(y_test_pred,
y_test_pred - y_test,
c='lightgreen',
marker='s',
s=35,
alpha=0.7,
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
