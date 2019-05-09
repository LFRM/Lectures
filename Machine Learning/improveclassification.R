#improve classification using lasso

# Consider the information in the Stock Market Data, available at ISLR.
# Let:
#    Y = Direction
#    X's = Lag1, Lag,2 Lag3, Lag4, Lag5 and Volume
#    Test data = data reported for year 2005
#    Threshold = 0.50

library(ISLR)
attach(Smarket)
train = (Year < 2005)
Smarket.2005 = Smarket[!train,]
Direction.2005 = Direction[!train]

#1. Logit

glm.fits = glm(Direction ~ Lag1 + Lag2, data = Smarket, family = binomial, subset = train) 
glm.probs = predict(glm.fits, Smarket.2005, type = "response")
glm.pred = rep("Down", 252)
glm.pred[glm.probs >.5] = "Up"
table(glm.pred, Direction.2005)
1 - mean(glm.pred == Direction.2005)

library(glmnet)
grid = 10^seq(10,-5, length = 10^3)
x = cbind(Lag1, Lag2)
y = Direction

#logit repeated
pen.fits = glmnet(x[train,], y[train], alpha = 0, lambda=0, family = "binomial")
pen.probs = predict(pen.fits, s = 0, newx = x[!train,], type = "response")
plot(glm.probs); lines(pen.probs, type = 'p', col = 2, pch = 3)

#rr : alpha = 0 
set.seed(1)
cv.out = cv.glmnet(x[train, ], y[train], alpha=0, lambda = grid, family = "binomial", type.measure = "class")
bestlam = cv.out$lambda.min
pen.fits = glmnet(x[train,], y[train], alpha = 0,  lambda = bestlam, family = "binomial")
rr.pred = predict(pen.fits, newx = x[!train,], s = "lambda.min", type = "class")

table(rr.pred, Direction.2005)
1 - mean(rr.pred == Direction.2005)

#lasso : alpha = 1
set.seed(1)
cv.out = cv.glmnet(x[train, ], y[train], alpha=1, lambda = grid,family = "binomial", type.measure = "class")
bestlam = cv.out$lambda.min
pen.fits = glmnet(x[train,], y[train], alpha = 1,  lambda = bestlam, family = "binomial")
rr.pred = predict(pen.fits, newx = x[!train,], s = "lambda.min", type = "class")

table(rr.pred, Direction.2005)
1 - mean(rr.pred == Direction.2005)






