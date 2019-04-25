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
                 
#2. LOOCV
# Use the boot library and compute the LOOCV for the three models in part 1.b

#3. K-fold CV
# Use the boot library and compute the K-fold cv for the three models in part 1.b

