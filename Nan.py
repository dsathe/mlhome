'''
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data
array([[ 1.,
2.,
3.,
4.],
[ 5.,
6.,
3.,
8.],
[ 10., 11., 12.,
4.]])
#imputer replaces NaN with mean of the given values
'''

import pandas as pd
df = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
#print(df)

size_mapping = {'XL' : 3 ,'L':2 ,'M':1}
df['size']=df['size'].map(size_mapping)
#print(df)

import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel']=df['classlabel'].map(class_mapping)
print(df)
