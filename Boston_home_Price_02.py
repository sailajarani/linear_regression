import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# To display all columns in a dataframes we need to use following code needs
pd.pandas.set_option('display.max_column',None)

#To read data from csv file
df = pd.read_csv('BostonHousing.csv')

print(df.info())
print(df.shape)
# To know how many columns null value
features_with_null = [feature for feature in df.columns if df[feature].isnull().sum()>1]
print(features_with_null)

print(df.head())
print(df.corr())
df1 = df.drop(['chas','dis','b'],axis=1) #I thought less dependecy on dependent variable.
X = df1.drop(['medv'],axis=1)
y = df1['medv']
# find number of unique values in a dependent or target column
#if more number of unique values then we go for linear regression
print(y.nunique())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=23)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
plt.scatter(y_test,y_pred)
plt.show()
print(X_test.size)
print(y_test.size)

#plt.scatter(X_test,y_test,color='lavender')
plt.plot(X_test,y_pred,color='pink')
plt.xticks()
plt.yticks()
plt.show()
df_output = pd.DataFrame({'actual':y_test,'predicted':y_pred})
print(df_output)
#To know R^2 value, if it is near to 1 then it is good model or if it is near to zero then it is worst model.
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
print('r^2 is: ',r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))
# Changing model using Standard scaler method.

scale = StandardScaler()
scale.fit(X)
X_scale = scale.transform(X)
X_scale_train,X_scale_test,y_scale_train,y_scale_test = train_test_split(X_scale,y,test_size=0.3,random_state=62)
lr.fit(X_scale_train,y_scale_train)
y_scale_predict = lr.predict(X_scale_test)
print(pd.DataFrame({'actual':y_scale_test,'predicted':y_scale_predict}))
print(lr.score(X_scale_train,y_scale_train))
print(lr.score(X_scale_test,y_scale_test))
print(mean_squared_error(y_scale_test,y_scale_predict))
print(np.sqrt(mean_squared_error(y_scale_test,y_scale_predict)))
