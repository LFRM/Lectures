#This question involves subset selection, for which you will need the leaps library.

#Consider the Hitters data set in the ISLR library. Find the best linear model for the response Salary, using 

#a) best subset selection (bic)
#b) forward stepwise selection (bic)
#c) backward stepwise selection (bic)
#d) k-fold cross validation

library(ISLR)
library(leaps)
Hitters=na.omit(Hitters)

#a, b, c
regfit.full = regsubsets(Salary~.,  data = Hitters, nvmax = 19)
regfit.fwd  = regsubsets (Salary~., data = Hitters, nvmax = 19, method = "forward")
regfit.bwd  = regsubsets (Salary~., data = Hitters, nvmax = 19, method = "backward")

regfull.summary = summary(regfit.full)
regfwd.summary  = summary(regfit.fwd)
regbwd.summary  = summary(regfit.bwd)

bicfull.min = which.min(regfull.summary$bic)
bicfwd.min  = which.min(regfwd.summary$bic)
bicbwd.min  = which.min(regbwd.summary$bic)

plot(regfull.summary$bic,xlab = "Number of Variables ", ylab = "BIC",type = 'l', col = 'red', lwd = 3)
lines(regfwd.summary$bic,col = 'green', type = 'l', lty = 2, lwd = 3)
lines(regbwd.summary$bic,col = 'blue',  type = 'l', lty = 4, lwd = 3)
points(bicfull.min, regfull.summary$bic[bicfull.min], col="red", cex = 4, pch = 20)
points(bicfwd.min, regfwd.summary$bic[bicfwd.min], col="green", cex = 3, pch = 20)
points(bicbwd.min, regbwd.summary$bic[bicbwd.min], col="blue", cex = 2, pch = 20)
legend('bottomright', c("best subset", "fwd selection", "bwd selection"), lwd = c(1,1,1), col = c(2,3,4))

#d

predict.regsubsets = function (object, newdata, id, ...){
    form  = as.formula(object$call[[2]])
    mat   = model.matrix(form, newdata)
    coefi = coef(object, id = id)
    xvars = names(coefi)
    mat[, xvars]%*%coefi
}

k = 10
set.seed (1)
folds = sample(1:k, nrow(Hitters), replace = TRUE)
cv.errors = matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))

for(j in 1:k){
    best.fit=regsubsets(Salary ~ ., data=Hitters[folds!=j,],nvmax =19)
    for(i in 1:19){ 
        pred=predict(best.fit,Hitters[folds==j,],id=i)
        cv.errors[j,i]=mean((Hitters$Salary[folds==j]-pred)^2)
    }
}

mean.cv.errors = apply(cv.errors ,2,mean)
kfold.min = which.min(as.numeric(mean.cv.errors))

plot(mean.cv.errors ,type='b')
points(kfold.min, mean.cv.errors[kfold.min], col="red", cex = 2, pch = 20)







