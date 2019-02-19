import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Getting the datasets using the pandas library
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

from sklearn.tree import DecisionTreeRegressor
X=df[['LSTAT']].values
y=df['MEDV'].values
tree=DecisionTreeRegressor(max_depth=3)
#tree.fit(X,y)

### Plotting the graph
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None
'''
sort_idx=X.flatten().argsort()
lin_regplot(X[sort_idx],y[sort_idx],tree)
plt.xlabel('% lower status of population [LSTAT]')
plt.show('Price in $1000 [MEDV]')
plt.show()
'''
from sklearn.cross_validation import train_test_split
#X=df.iloc[:,:-1].values
y=df['MEDV'].values
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=1)
tree=DecisionTreeRegressor(max_depth=2)
tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
#plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
#plt.xlim([-10, 50])
plt.show()

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))


from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
