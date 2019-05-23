####################
#Lab: Decision Trees
####################

library(tree)
library(ISLR)
library(MASS)
library(gbm)

library(randomForest)

attach(Carseats)

# 1. Build a binary variable that is "Yes" if Sales >8, and "No" otherwise; and merge this new variable with the Carseat data in a new dataframe

High = ifelse(Sales <= 8, "No", "Yes")
Carseats = data.frame(Carseats ,High)

# 2. Build a classification tree using the new variable as response and all variables in the data set as predictors

tree.carseats = tree(High~. -Sales, Carseats)
summary(tree.carseats)

# 3. Use the plot() function to display the tree structure, and the text() function to display the node labels.

plot(tree.carseats)
text(tree.carseats, pretty = 1)

# 4. estimate the test error by splitting the observations into a training set and a test set as

#set.seed(2)
#train = sample(1:nrow(Carseats), 200)
#Carseats.test = Carseats[-train, ]
#High.test = High[-train]

# compute the predictions for the test set and write the corresponding confusion table

tree.carseats = tree(High~. -Sales, Carseats, subset = train)
tree.pred = predict(tree.carseats, Carseats.test, type = "class")
table(tree.pred ,High.test)

# 5. Use the function cv.tree() to perform cross-validation and determine the optimal level of tree complexity (k = alpha). Use FUN = prune.misclass  to indicate that we want the classification error rate to guide the cross-validation

set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)

# 6. Find the tree with the lowest cross-validation error rate. How many terminal nodes does this tree have?

# 50 errors
# 9 nodes

# 7. Plot the error rate as function of size and k

plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b") 

# 8. Apply the prune.misclass() function in order to prune the tree to obtain the nine-node tree.

prune.carseats = prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)

# 9. How well does this pruned tree perform on the test data set? Use the predict function and build a confusion matrix

tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)

# 10. Fit a regression tree of medv, explained by all variables in the Boston data set. Use the following train set

#set.seed(1)
#train = sample(1:nrow(Boston), nrow(Boston)/2)

tree.boston = tree(medv~., Boston, subset = train)
summary(tree.boston)

plot(tree.boston)
text(tree.boston, pretty = 0)

# 11. Find the tree with the lowest cross-validation error rate. How many terminal nodes does this tree have?
cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = 'b')

prune.boston = prune.tree(tree.boston, best = 7)
plot(prune.boston)
text(prune.boston, pretty = 0)

# 12. Evaluated the optimal tree from cv and compute the test MSE

yhat = predict(tree.boston, newdata = Boston[-train, ])
boston.test = Boston[-train, "medv"]

plot(yhat,boston.test)
abline(0,1)
mean((yhat - boston.test) ^ 2)

# 13. Build a bagging model for medv as function of all the variables in the Boston data set

set.seed(1)
bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = 13, importance = TRUE)
bag.boston

# 14. Evaluate the bagging model in the test set. Compute the test MSE and compared it with the one in question 12.

yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)

# 15. Evaluate if setting the ntree argument in randomForest, the test MSE improves

bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = 13, ntree = 25)
yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test) ^ 2)

# 16. Repeat question 13, but using a random forest instead. Use floor(sqrt(p)) as number of variables to be considered in each split.

set.seed(1)
rf.boston = randomForest(medv~., data = Boston, subset = train, mtry = floor(sqrt(13)),importance = TRUE)
yhat.rf = predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test) ^ 2)

# 17. Display  the importance of each variable using the function importance(), and plot it using varImpPlot()

importance (rf.boston)
varImpPlot(rf.boston)

# 18. Repeat question 13 with boosting instead

set.seed (1)
boost.boston = gbm(medv~., data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
summary(boost.boston)

#19. use the boosted model to predict medv on the test set. Compare the results with those of bagging and random forest

yhat.boost = predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test) ^ 2)

# 20. Repeat 18 with a shrinkage parameter of 0.2

boost.boston=gbm(medv~., data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage  = 0.2, verbose = F)
yhat.boost=predict(boost.boston, newdata = Boston[-train,], n.trees = 5000)
mean((yhat.boost - boston.test) ^ 2)











