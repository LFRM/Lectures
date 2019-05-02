#Consider the Hitters data in the ISLR library.

#a) Use the glmnet function in the glmnet library to fit a ridge regression for lambda = 0.7, and compare this with the results from lm.

library(ISLR)
library(glmnet)

x = model.matrix(Salary~.,Hitters)[,-1]
y = Hitters$Salary
ridge.mod = glmnet(x, y, alpha=0, lambda = 0.7)

ridge.mod$beta
lm(Salary~.,Hitters)$coefficients

#b) select the best lambda with the validation set approach, using half of the observations for training and half for test. Report the test MSE associated with this value of λ

set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]

cv.out = cv.glmnet(x[train, ], y[train], alpha=0)
bestlam = cv.out$lambda.min
bestlam

ridge.pred = predict(ridge.mod, s=bestlam, newx = x[test,])
mean((ridge.pred-y.test)^2)

#c) re-estimate the ridge regression model with the lambda found in part b
ridge.mod = glmnet(x, y, alpha=0, lambda = bestlam)

#d) repeat part a) using lasso instead of ridge regression
lasso.mod = glmnet(x, y, alpha = 1, lambda = 0.7)

