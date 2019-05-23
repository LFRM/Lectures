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

# 2. Build a classification tree using the new variable as response and all variables in the data set as predictors

# 3. Use the plot() function to display the tree structure, and the text() function to display the node labels.

# 4. estimate the test error by splitting the observations into a training set and a test set as

#set.seed(2)
#train = sample(1:nrow(Carseats), 200)
#Carseats.test = Carseats[-train, ]
#High.test = High[-train]

# compute the predictions for the test set and write the corresponding confusion table

# 5. Use the function cv.tree() to perform cross-validation and determine the optimal level of tree complexity (k = alpha). Use FUN = prune.misclass  to indicate that we want the classification error rate to guide the cross-validation

# 6. Find the tree with the lowest cross-validation error rate. How many terminal nodes does this tree have?

# 7. Plot the error rate as function of size and k

# 8. Apply the prune.misclass() function in order to prune the tree to obtain the nine-node tree.

# 9. How well does this pruned tree perform on the test data set? Use the predict function and build a confusion matrix


# 10. Fit a regression tree of medv, explained by all variables in the Boston data set. Use the following train set

#set.seed(1)
#train = sample(1:nrow(Boston), nrow(Boston)/2)

# 11. Find the tree with the lowest cross-validation error rate. How many terminal nodes does this tree have?

# 12. Evaluated the optimal tree from cv and compute the test MSE

# 13. Build a bagging model for medv as function of all the variables in the Boston data set

# 14. Evaluate the bagging model in the test set. Compute the test MSE and compared it with the one in question 12.

# 15. Evaluate if setting the ntree argument in randomForest, the test MSE improves

# 16. Repeat question 13, but using a random forest instead. Use floor(sqrt(p)) as number of variables to be considered in each split.

# 17. Display  the importance of each variable using the function importance(), and plot it using varImpPlot()

# 18. Repeat question 13 with boosting instead

#19. use the boosted model to predict medv on the test set. Compare the results with those of bagging and random forest

# 20. Repeat 18 with a shrinkage parameter of 0.2











