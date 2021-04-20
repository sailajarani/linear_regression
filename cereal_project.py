import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
import numpy as np

pd.pandas.set_option('display.max_column',None) # To display all column values in dataset
df = pd.read_csv('cereal.csv')
print(df.head())
print(df.info())
print(df.shape)

# Trying to find what are unique values in 'type' and "mfr' columns in df data set
print(df.type.unique())
print(df.mfr.unique())

# Changing categorical input column to numeric values using map() function.
df['type']=df['type'].map({'C':0,'H':1})
df['mfr'] = df['mfr'].map({'N':0,'Q':1,'K':2,'R':3,'G':4,'P':5,'A':6})
print(df.head())

# Find the correlation of input features on output
sns.heatmap(df.corr())
print(df.corr())
plt.show()

# Dropping less correlation input features 'name','carbo','shelf' along output coulmn 'rating'
X = df.drop(['rating','name','carbo','shelf'],axis=1)
# Rating is output or dependent feature
y = df['rating']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=12)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
diff_data = pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(diff_data.head())

# plotting for test ouput to predicted output
plt.scatter(y_test,y_pred)

plt.show()

# Performance checking in LinearRegression
print('Mean_squared_error:  ',mean_squared_error(y_test,y_pred))
print('mean_absolute_error:  ',mean_absolute_error(y_test,y_pred))
print('Root_mean_squared_error:  ',np.sqrt(mean_squared_error(y_test,y_pred))) # Less value of root_mean_squared is indicated that it is good model.
print('r2_score: ',r2_score(y_test,y_pred))   #If r2_score greter than 0.7 then it is good model.


