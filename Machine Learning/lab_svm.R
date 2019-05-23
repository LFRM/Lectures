#############################
#Lab: Support Vector Machines
#############################

library(ISLR)
library(e1071)
library(ROCR)


# 1. generate  training data using

#set.seed (1)
#x = matrix(rnorm(20 * 2), ncol = 2)
#y = c(rep(-1,10), rep(1,10))
#x[y == 1,] = x[y == 1, ] + 1
#dat = data.frame(x = x, y = as.factor(y))

# and test data using

#xtest = matrix(rnorm(20 * 2), ncol = 2)
#ytest = sample(c(-1,1), 20, rep = TRUE)
#xtest[ytest == 1, ] = xtest[ytest == 1, ] + 1
#testdat = data.frame(x = xtest, y = as.factor(ytest))

# Is the data linearly separable?

# 2.  Use the svm()  function from the library(e1071) to fit the support vector classifier for different values of the cost parameter. Note: this implementation is not exactly what we saw in class. Consider the cost parameter in svm() as the cost of the violation of the margin. 

# 3. Plot the resulting decision boundary. What do the colors represent? What do the crosses represent?

# 4. Print the values corresponding to the support vector

# 5. Select the optimal cost parameter using the tune() function from the  library(e1071) using 10-fold cross validation

# 6. Use the predict() function to build a confusion matrix

# 7. Modify the training data to make it separable using
#x[y == 1,] = x[y == 1,] + 0.5

# Fit the support vector classifier and plot the resulting hyperplane, using a very large value of cost so that no observations are misclassified

# 8. generate non-linear data as

#set.seed (1)
#x = matrix(rnorm(200*2), ncol = 2)
#x[1:100,] = x[1:100,] + 2
#x[101:150,] = x[101:150,] - 2
#y = c(rep(1,150), rep(2,50))
#dat = data.frame(x = x, y = as.factor(y))
#train = sample(200,100)
#plot(x[,c(2,1)], col=y)

# Fit an svm model using a radial kernel with parameter gamma = 1 and cost = 1

# 9. Refit the model with cost = 1e5. What happens with the fit?

# 10. Select the optimal cost parameter, and optimal gamma using the tune() function from the  library(e1071) using 10-fold cross validation

# 11. Use the predict() function to build a confusion matrix

# 12. Plot ROC curves comparing a model with gamma=2, and gamma =50. How is the behavior for training data and for test data?
                                                                                                                          
# 13. generate non-linear with three categories  as

#set.seed (1)
#x = rbind(x, matrix(rnorm(50 * 2), ncol = 2))
#y = c(y, rep(0, 50))
#x[y == 0, 2] = x[y == 0, 2] + 2
#dat = data.frame(x = x, y = as.factor(y))
#par(mfrow = c(1,1))
#plot(x[,c(2,1)], col = (y + 1))

# and fit an svm model 


















