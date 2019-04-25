#########################################
#Lab: Resampling Methods
#########################################

# Consider the information in the Auto Data, available at ISLR.
# Let:
#    Y = mpg
#    X's = horsepower

#1. Validation Set Approach
#a. Create an train index vector of 196 elements. Call it "train" 
#   
#b. Compute the test MSE for three models:
#    mpg vs horsepower
#    mpg vs horsepower^2
#    mpg vs horsepower^3
#    Which one has the better MSE? What if we re-run the computations?

library(ISLR)
set.seed(1)
train = sample( 392, 196 )
attach(Auto)

set.seed(1)
cv.error = rep(0,3)
for (i in 1:3){
    lm.fit = lm( mpg ~ poly( horsepower , i ), data = Auto, subset = train )
    cv.error[i] = mean( ( mpg - predict( lm.fit, Auto ) )[-train] ^ 2 )
}

#2. LOOCV
# Use the boot library and compute the LOOCV for the three models in part 1.b

library(boot)
set.seed(2)
cv.error = rep(0,3)
for (i in 1:3){
    glm.fit = glm( mpg ~ poly( horsepower , i ), data = Auto)
    cv.error[i] = cv.glm( Auto , glm.fit )$delta[1]
}


#3. K-fold CV
# Use the boot library and compute the K-fold cv for the three models in part 1.b

set.seed(3)
cv.error = rep(0,3)
for (i in 1:3){
    glm.fit = glm( mpg ~ poly( horsepower , i ), data = Auto)
    cv.error[i] = cv.glm( Auto , glm.fit,K=10)$delta[1]
}
