import pandas as pd
import numpy as np

### Getting the datasets using the pandas library
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()
### Getting the input and output in X, y  matrix respectively
X = df[['RM']].values
y = df['MEDV'].values

### using the StandardScaler to preprocess the matrices
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1))

### using the sklearn linearlregression model
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
#print('Slope: %.3f' % slr.coef_[0])

### Plotting the graph
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

### using the matplot library for plotting the data
import matplotlib.pyplot as plt
lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
