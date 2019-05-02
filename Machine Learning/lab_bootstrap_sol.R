#Consider the "portfolio" data in the ISLR library

#a) Write a function that computes the optimal weights as presented in class.
library(ISLR)
alpha.fn = function(data, index){
    X = data$X[index]
    Y = data$Y[index]
    return((var(Y) - cov(X,Y)) / (var(X) + var(Y) - 2*cov(X,Y)))
}
alpha.fn(Portfolio,1:100)
alpha.fn(Portfolio,sample(100,100,replace=T))

#b) Produce  1000 bootstrap estimates for alpha. What is the SE of the alpha estimate? Use the boot function from the boot library.

library(boot)
bootstrap = boot(Portfolio, alpha.fn, R = 1000)
