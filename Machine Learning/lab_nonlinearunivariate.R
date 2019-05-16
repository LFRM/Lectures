#############################
# Lab: non-linear / univariate
#############################

#In this lab we use the Wage data from the ISLR library.

# 1. Write a model of wage as a fourth degree polynomial function of age.

# 2. Use the function "poly" inside lm() to answer 1 using a) raw = F; b) raw = T. Plot the predictors in each case. 

# 2.1 Are the models the same? 
                                        
# 2.2. Are the predictions the same? Compute the predictions for the grid

# agelims = range(age)
# age.grid = seq(from = agelims[1], to = agelims[2])

# 2.3 Plot the predictions for obs age.grid

# 3. Do we really need a 4th degree polynomial in the model? Use the anova function to check. 

# 4. Consider the response

# y = 1, if wage > 250 
# y = 0, if wage <= 250

# and poly(age, 4) as the predictors. Make a plot of the predictions in the grid age.grid

# 5. Fit wages using a discretized version of age, by using cut(age,4) 

# 6. For the next questions we use the library splines

# 6.1. Fit wage using regression splines  using the b-splines (bs function) for knots located at 25, 40 and 60. Plot the results. Compute the degrees of freedom. 

# 6.2. Fit wage using natural splines (ns function) for knots located at 25, 40 and 60. 

# 6.3. Fit wage using smoothing splines  (smooth.spline function). compare the case df = 16 with cv = T.

#7. Fit wage using local regression via loess. Compare the results using span =0.5 and span = 0.2

