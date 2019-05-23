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
plot(x[,c(2,1)], col = (3 - y))

# 2.  Use the svm()  function from the library(e1071) to fit the support vector classifier for different values of the cost parameter. Note: this implementation is not exactly what we saw in class. Consider the cost parameter in svm() as the cost of the violation of the margin. 

svmfit.01 = svm(y ~., data = dat, kernel = "linear", cost = 0.1, scale = FALSE)
svmfit.10 = svm(y ~., data = dat, kernel = "linear", cost = 10, scale = FALSE)
summary(svmfit.10)

# 3. Plot the resulting decision boundary. What do the colors represent? What do the crosses represent?

plot(svmfit.01, dat)
plot(svmfit.10, dat)

# 4. Print the values corresponding to the support vector

svmfit.01$index
svmfit.10$index

# 5. Select the optimal cost parameter using the tune() function from the  library(e1071) using 10-fold cross validation

set.seed(1)
tune.out = tune(svm, y~., data = dat, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1,5, 10, 100)))
summary(tune.out)

bestmod = tune.out$best.model
summary(bestmod)

# 6. Use the predict() function to build a confusion matrix

ypred = predict(bestmod ,testdat)
table(predict = ypred, truth = testdat$y)

# 7. Modify the training data to make it separable using
#x[y == 1,] = x[y == 1,] + 0.5
#plot(x[,c(2,1)], col = (y + 5) / 2, pch = 19)

# Fit the support vector classifier and plot the resulting hyperplane, using a very large value of cost so that no observations are misclassified

dat = data.frame(x = x, y = as.factor(y))
svmfit = svm(y ~., data = dat, kernel = "linear", cost = 1e5)
summary(svmfit)

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

svmfit = svm(y~., data=dat[train,], kernel = "radial", gamma = 1,cost = 1)
summary(svmfit)

plot(svmfit, dat[train,])

# 9. Refit the model with cost = 1e5. What happens with the fit?

svmfit = svm(y~., data = dat[train, ], kernel = "radial", gamma = 1, cost = 1e5)
plot(svmfit, dat[train,])

# 10. Select the optimal cost parameter, and optimal gamma using the tune() function from the  library(e1071) using 10-fold cross validation

set.seed (1)
tune.out = tune(svm, y~., data = dat[train, ], kernel = "radial", ranges = list(cost = c(0.1,1,10,100,1000), gamma = c(0.5,1,2,3,4) ))
summary(tune.out)

bestmod = tune.out$best.model
summary(bestmod)

# 11. Use the predict() function to build a confusion matrix

table(true = dat[-train,"y"], pred = predict(tune.out$best.model, newdata = dat[-train, ]))

# 12. Plot ROC curves comparing a model with gamma=2, and gamma =50. How is the behavior for training data and for test data?
                                                                                                                          
rocplot = function(pred, truth, ...){
    predob = prediction (pred, truth)
    perf = performance (predob , "tpr", "fpr")
    plot(perf ,...)
}

svmfit.opt = svm(y~., data = dat[train,], kernel="radial", gamma = 2, cost = 1, decision.values = T)
fitted = attributes(predict(svmfit.opt, dat[train, ], decision.values = TRUE))$decision.values

par(mfrow=c(1,2))
rocplot(fitted, dat[train ,"y"], main = "Training Data")

svmfit.flex = svm(y~., data = dat[train, ], kernel = "radial", gamma = 50, cost = 1, decision.values = T)
fitted = attributes(predict(svmfit.flex, dat[train, ], decision.values = T))$decision.values
rocplot(fitted, dat[train, "y"], add = T, col = "red")

fitted = attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"], main = "Test Data")
fitted = attributes(predict(svmfit.flex, dat[-train, ], decision.values = T))$decision.values
rocplot(fitted, dat[-train, "y"],add = T, col = "red")

# 13. generate non-linear with three categories  as

#set.seed (1)
#x = rbind(x, matrix(rnorm(50 * 2), ncol = 2))
#y = c(y, rep(0, 50))
#x[y == 0, 2] = x[y == 0, 2] + 2
#dat = data.frame(x = x, y = as.factor(y))
#par(mfrow = c(1,1))
#plot(x[,c(2,1)], col = (y + 1))

# and fit an svm model 

svmfit = svm(y~., data = dat, kernel = "radial", cost = 10, gamma = 1)
plot(svmfit, dat)

















