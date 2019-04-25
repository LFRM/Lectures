#########################################
#Lab: Classification
#########################################

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
#a. Use the glm function and compute a logit model of
#   Direction vs Lag1 + Lag2
#b. Using the model in part a, predict  "direction" for the test set. What is the error rate?

glm.fits = glm(Direction ~ Lag1 + Lag2, data = Smarket, family = binomial,subset = train)
glm.probs = predict(glm.fits, Smarket.2005, type = "response")
glm.pred = rep("Down", 252)
glm.pred[glm.probs >.5] = "Up"
table(glm.pred, Direction.2005)
1 - mean(glm.pred == Direction.2005)

#2. LDA
#a. Use the lda function and Compute an LDA model of
#   Direction vs Lag1 + Lag2
#b. Using the model in part a, predict  "direction" for the test set. What is the error rate?

library(MASS)
lda.fit = lda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
lda.class = predict(lda.fit, Smarket.2005)$class
table(lda.class, Direction.2005)
1-mean(lda.class == Direction.2005)


#3. QDA
#a. Use the qda function and compute a QDA model of
#   Direction vs Lag1 + Lag2
#b. Using the model in part a, predict  "direction" for the test set. What is the error rate?

qda.fit = qda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
qda.class = predict(qda.fit,Smarket.2005)$class
table(qda.class, Direction.2005)
1- mean(qda.class == Direction.2005)

#4. KNN
#a. Use the knn function and compute a KNN (k=3) model for:
#   Direction vs Lag1 + Lag2
#b. Using the model in part a, predict  "direction" for the test set. What is the error rate?

library(class)
train.X = cbind(Lag1 ,Lag2)[train,]
test.X = cbind(Lag1,Lag2)[!train,]
train.Direction = Direction [train]

set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction, k=3)
table(knn.pred,Direction.2005)
1- mean(knn.pred == Direction.2005)
