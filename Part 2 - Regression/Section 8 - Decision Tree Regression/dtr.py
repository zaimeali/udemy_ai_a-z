# Decision Tree Regression

# Lib Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Data Set Import
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

# Plot Decision tree plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Decision Tree Regressor")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()

# DTR is non-continous algo