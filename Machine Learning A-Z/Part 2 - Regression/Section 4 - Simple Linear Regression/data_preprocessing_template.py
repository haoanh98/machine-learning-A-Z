# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('salary_data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


plt.scatter(x_train,y_train, color = 'red' )
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('train')
plt.xlabel('Nam kinh nghiem')
plt.ylabel('luong')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred, color='blue')
plt.title('test')
plt.xlabel('Nam kinh nghiem')
plt.ylabel('luong')
plt.show()