#################################
# Lab: non-linear / multivariate
#################################

# In this lab we use the Wage data from the ISLR library.
# We use the gam library

library(gam)

#1. Consider  a model M0 of wage as function 

#linear in education
#smoothing spline with 5 degrees of freedom in age

#use ANOVA to compare between the following nested models:

# 1) model M0 
# 2) model M0 with  years added as a linear predictor
# 3) model M0 with years added as  a spline function with 5 df.

# which model should be used?

gam.m1 = gam(wage ~ s(age, df = 5) + education, data = Wage)
gam.m2 = gam(wage ~ year + s(age, df = 5) + education, data = Wage)
gam.m3 = gam(wage ~ s(year, df = 4) + s(age, df = 5) + education, data = Wage)

anova(gam.m1, gam.m2, gam.m3, test="F")

#=> gam.m2 is the best model. 

preds = predict(gam.m2, newdata = Wage)

# 2. Consider the best model from part 1. Add an interaction effect age*year using local regression with the lo() function and a span of 0.2. 

gam.lo.i = gam(wage ~ lo(year, age, span = 0.2) + education, data = Wage)

# 3. Consider the response

# y = 1, if wage > 250 
# y = 0, if wage <= 250

# and fit a model using as predictors year, education and s(age, df = 5). PLot the fit of the model. 

gam.lr = gam(I(wage>250) ~ year + s(age,df=5) + education, family = binomial, data = Wage)
par(mfrow = c(1,3)); plot(gam.lr, se = T, col = "green")





