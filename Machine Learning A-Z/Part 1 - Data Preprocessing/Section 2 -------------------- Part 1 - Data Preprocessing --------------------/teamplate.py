import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
x[:,1:3]=imputer.fit_transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,0]=labelencoder_X.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

y=labelencoder_X.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)