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

# no need of feature scaling bcz of PLR

# Linear Model
from sklearn.linear_model import LinearRegression
SLReg = LinearRegression()
SLReg.fit(X, y)


# Polynomial Linear Model
from sklearn.preprocessing import PolynomialFeatures
PLReg = PolynomialFeatures(degree = 4) # i changed it from 2 to 3 then 3 to 4 then 4 to 5 to check blue line touches red dots
# now again changing it to 4 to 5
X_Poly = PLReg.fit_transform(X)
# 1s column in X_poly is bcz we manually add that column in MLR but here it was add automatically
SLReg2 = LinearRegression()
SLReg2.fit(X_Poly, y)

# Visualizing LR result
plt.scatter(X, y, color = 'red')
plt.plot(X, SLReg.predict(X), color = 'blue')
plt.title("Truth or Bluff ~ Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing PLR result
plt.scatter(X, y, color = 'red')
plt.plot(X, SLReg2.predict(PLReg.fit_transform(X)), color = 'blue')
plt.title("Truth or Bluff ~ Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


