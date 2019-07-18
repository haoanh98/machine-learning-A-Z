import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:, -1].values

'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
x[:,1:3]=imputer.fit_transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,0]=labelencoder_X.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

y=labelencoder_X.fit_transform(y)'''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

result=classifier.score(x_test,y_test)

from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2 = np.meshgrid(np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01),
                    np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.4,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c = ListedColormap(('red','green'))(i),label=j)
plt.title('KNN(train)')
plt.xlabel('Age')
plt.ylabel('salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2 = np.meshgrid(np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01),
                    np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01))
y_d=np.array([x1.ravel(),x2.ravel()]).T
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.4,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c = ListedColormap(('red','green'))(i),label=j)
plt.title('KNN(test)')
plt.xlabel('Age')
plt.ylabel('salary')
plt.legend()
plt.show()
      