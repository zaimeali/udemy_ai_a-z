'''
R Squared:
	if SSresidual is 0 then RSquare will be an ideal state .
	R Square can be -ve also
	R square range 0 to 1 => 0 worse 1 better
	
'''

# R^2 = 1 - (SSresidual/SSavg)

# R Square will be not good choice for MLR because of multiple variables.

# Adjusted R Square 
# AdjR^2 = 1 - (1-R^2)(n-1)/(n-p-1)
# p = no. of regressors (variables)
# n = sample size 
# AdjR^2 is good for MLR

