# Support Vector Regression

'''
SVR support linear and non linear regression

SVR performs linear regression in a higher (dimensional space)

each data point in the training represents it's own dimension.

resulting value gives you the coordinate of your test point in that dimension

The vector we get when we evaluate the test point  for all  points in the training set,
k is the representation of the test point in the higher dimensional space.

Once you have that vector you can use it to perform a Linear Regression

# SVR Requirements:
->It requires a training set T = {X, Y} which covers the domain of interest and is accompanied by 
	solutions of that domain
->The work of the SVM is to approximate the function we used to generate the training set
	F(X)=Y
	
In a Classification problem, vector X are used to define a hyperplane that seperates the 2 different
	classes in your solution

These vectors are used to perform Linear Regression. The vector closest to the test point 
are referred to as support vectors. We can evaluate our function anywhere so any vectors could be
closest to our test evaluation location.
'''


