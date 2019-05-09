#Consider the Hitters data in the ISLR library.

#a) Use the glmnet function in the glmnet library to fit a ridge regression for lambda = 0.7, and compare this with the results from lm.

library(ISLR)
library(glmnet)

Hitters = na.omit(Hitters)

x = model.matrix(Salary ~., Hitters)[,-1]
y = Hitters$Salary
grid = 10^seq(10, -2, length = 100)

glmnet(x, y, alpha = 0, lambda = 0.7)$beta
lm(Salary ~., Hitters)$coefficients

#b) select the best lambda with the validation set approach, using half of the observations for training and half for test. 

set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]

ridge.mod = glmnet(x[train,], y[train], alpha = 0, lambda = grid)

cv.out = cv.glmnet(x[train, ], y[train], alpha = 0, lambda = grid)
bestlam = cv.out$lambda.min
bestlam


#c) Report the test MSE associated with the  value of λ found in part b
ridge.pred = predict(ridge.mod, s = bestlam, newx = x[test,])
mean((ridge.pred - y.test)^2)


#d) repeat part a), b and c) using lasso instead of ridge regression

glmnet(x, y, alpha = 1, lambda = 0.7)$beta
lm(Salary ~., Hitters)$coefficients

set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]

lasso.mod = glmnet(x[train,], y[train], alpha = 1, lambda = grid)

cv.out = cv.glmnet(x[train, ], y[train], alpha = 1)
bestlam = cv.out$lambda.min
bestlam

lasso.pred = predict(lasso.mod, s = bestlam, newx = x[test,])
mean((lasso.pred - y.test)^2)


