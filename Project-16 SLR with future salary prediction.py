# SLR with future salary prediction

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# call dataset and split data into iv and dv
ds = pd.read_csv(r'D:\1. Professionall\Data Science\08-11-2023\Salary_Data.csv')

X = ds.iloc[:, :-1].values
y = ds.iloc[:, 1].values

# Lets split the data into 80-20%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

# as the dataset has 2 attributes we use SLR algorithm

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Built regression model
regressor.fit(X_train, y_train)

# Test the model and prepare future prediction table
y_pred = regressor.predict(X_test)

# visualize train data point ( 24 data)
plt.scatter(X_train, y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set) 75-25%')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visulaize test data point 
plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Testing set) 75-25%')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# To get value of slope
m = regressor.coef_
m

# To get intercept or constant  
c = regressor.intercept_
c

# predict or forcast the future the data which we not trained before 
y_12 = int(m) * 12 + c
y_12

y_20 = int(m) * 20 + c
y_20


# to check overfitting  ( low bias high variance)
bias = regressor.score(X_train, y_train)
bias


# to check underfitting (high bias low variance)
variance = regressor.score(X_test,y_test)
variance







