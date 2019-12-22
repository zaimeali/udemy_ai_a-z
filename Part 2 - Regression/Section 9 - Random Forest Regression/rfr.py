# Random Forest Regression

'''
STEPS:
	1) Pick at random K data points from the training set
	2) Build the decision tree associated to these K data points.
	3) Choose the number NTree of trees you want to build and repeat STEPS 1 & 2.
	4) For a new data point, make each one of your NTree trees predict the value of Y to
		for the data point in Question, and assign the new data point the avg across all of
		the predicted Y values.
'''

# Random forest algo doesn't handle outliers specifically

# import Lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1: -1].values
y = data.iloc[:, -1].values

# Random Forest Lib
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) # creating 300 trees
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

# Random Forest is non-continous model

# Visualize
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff ~ Random Forest Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
