import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
pd.pandas.set_option('display.max_column',None)

df = pd.read_csv('FyntraCustomerData.csv')
print(df.head())
print(df.shape)
print(df.info())
# columns 'Email',Address' have number of unique object so delete those columns in dataframe.
df = df.drop(columns=['Email','Address'])
print(df.info())
print(df.corr()) # Yearly amount spent depends less on Time_on_website, So we can delete the column.

# Check number of unique values in Avatar.
print(df.Avatar.nunique())  # It has 138 unique values so delete the column.
df.drop(['Avatar','Time_on_Website'],axis=1,inplace=True)
print(df.info())

# spilt the data for input and output
X = df.drop(columns=['Yearly_Amount_Spent'])
y = df['Yearly_Amount_Spent']

# dependent column have number of numeric values so it is continuous values so we have to use Linear regression.

lr = LinearRegression()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2,test_size=0.2)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

output = pd.DataFrame({'Actual':y_test,'predict':y_pred})

# graph between actual and predict
plt.scatter(y_test,y_pred)
plt.show()

# performance check

print('mean_absolute_error: ', mean_absolute_error(y_test,y_pred))
print('mean_square_error: ', mean_squared_error(y_test,y_pred))
print('root_mean_squared_error: ', np.sqrt(mean_squared_error(y_test,y_pred)))
print("r^2 score: ", r2_score(y_test,y_pred))