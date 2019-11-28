# Multiple Linear Regression

# y = b0 + b1*X1 + b2*X2+.....+bn*Xn

# Must read Help_MLR.txt

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Slicing 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap
X = X[:, 1:]


#Splitting the data
from sklearn.model_selection import train_test_split
# 20% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)