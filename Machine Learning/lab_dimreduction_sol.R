#Consider the Hitters data in the ISLR library.

#a) use the pls library and compute PCR to explain Salary in terms of all variables in the data set. Select scale = TRUE to scale the variables and validation = "CV". 

library(pls)
set.seed (2)
pcr.fit = pcr(Salary~., data = Hitters , scale = TRUE, validation = "CV")

#b) How many components do we need to explain at least 90% of the variance in the design matrix?
summary(pcr.fit)

#c) Repeat part b) using pls. Do we need the same number of components?
set.seed(1)
pls.fit=plsr(Salary~., data = Hitters, scale=TRUE,  validation ="CV")
summary(pls.fit)















