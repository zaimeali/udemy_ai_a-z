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


# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting
y_pred = regressor.predict(X_test)


# Stats for backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# Optimal matrix X will only contain the independent variables that have high impact
X_Opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_Opt).fit()
# Step 2 Done in Above


# Summary Function => 
regressor_OLS.summary() # P-Value > Significant Value, so remove predictor



# Now Starting again to remove some variables => [2]
X_Opt = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_Opt).fit()
# Step 2 Done in Above

# Summary Function => 
regressor_OLS.summary() # P-Value > Significant Value, so remove predictor



# Now Starting again to remove some variables => [3]
X_Opt = X[:, [0, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_Opt).fit()
# Step 2 Done in Above

# Summary Function => 
regressor_OLS.summary() # P-Value > Significant Value, so remove predictor


# Now Starting again to remove some variables => [3]
X_Opt = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_Opt).fit()
# Step 2 Done in Above

# Summary Function => 
regressor_OLS.summary() # P-Value > Significant Value, so remove predictor


# Now Starting again to remove some variables => [3]
X_Opt = X[:, [0, 3]]

regressor_OLS = sm.OLS(endog = y, exog = X_Opt).fit()
# Step 2 Done in Above

# Summary Function => 
regressor_OLS.summary() # P-Value > Significant Value, so remove predictor
# const in summary is columns index

# According to this only one independent variable that is R&D

