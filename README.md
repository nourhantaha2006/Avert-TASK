# Avert-TASK


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
pd.read_csv(r'C:\Users\I SEvEN\Desktop\Tasks of data\dataset\data tasks\advertising (1).csv')
df = pd.read_csv(r'C:\Users\I SEvEN\Desktop\Tasks of data\dataset\data tasks\advertising (1).csv')
print (df.isnull().sum())
df.fillna(df.mean(), inplace=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['TV']] = scaler.fit_transform(df[['TV']])

from sklearn.model_selection import train_test_split
X=df.drop('Sales',axis=1)
y=df['Sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,3))
sns.lineplot(x= 'TV',y='Sales', data=df)
plt.title( 'Amount of sales')
plt.show()

plt.figure(figsize=(5,3))
sns.lineplot(x= 'Radio',y='Sales', data=df)
plt.title( 'Amount of sales')
plt.show()

plt.figure(figsize=(5,3))
sns.lineplot(x= 'Newspaper',y='Sales', data=df)
plt.title( 'Amount of sales')
plt.show()



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

import numpy as np

y_pred = np.round(y_pred) 
y_pred = y_pred.astype(int)  
print(set(y_test))  

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
