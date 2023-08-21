# Importing liberies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data and divide into dv & iv

dataset = pd.read_csv(r'D:\1. Professionall\Data Science\08-09-2023\Data.csv')

X = dataset.iloc[:, :-1].values	

y = dataset.iloc[:,3].values

# Filling null values in X dataset

from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy="median") # default startegy is mean 

impute = impute.fit(X[:,1:3]) 
X[:, 1:3] = impute.transform(X[:,1:3]) 

# Convert categorical data into numerical data

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Spliting data into training set and testing set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)









