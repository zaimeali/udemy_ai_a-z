# PLR is a special case of MLR

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import dataset
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, -2: -1].values
y = data.iloc[: , -1].values

# split dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# we will not split data bcz of less data



# Regression Model

""" Create Your Regressor Here """

# Predict PLR
y_pred = regressor.predict([[6.5]])

# Visualizing PLR result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Truth or Bluff ~ Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing PLR result for Higher Res
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff ~ Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

