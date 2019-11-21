# Simple Linear Regression

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Split the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting SLR to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting 
# create a vector that will contain the prediction values
y_pred = regressor.predict(X_test)

# Visualizing
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience ~ Training Set Result")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #bcz our model is built on X_train and y_pred is actually to compare it with y_test
plt.title("Salary vs Experience ~ Test Set Result")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()