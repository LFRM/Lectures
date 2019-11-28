#############################
# Lab: non-linear / univariate
#############################

#In this lab we use the Wage data from the ISLR library.

library(ISLR)
attach(Wage)

# 1. Write a model of wage as a fourth degree polynomial function of age.

# => many alternatives:
age1 = age;
age2 = age ^ 2;
age3 = age ^ 3;
age4 = age ^ 4;

fita = lm(wage ~ age1 + age2 + age3 + age4,data = Wage)
fitb = lm(wage ~ cbind(age, age ^ 2, age ^ 3, age ^ 4), data = Wage)
fitc = lm(wage ~ age + I(age ^ 2) + I(age ^ 3) + I(age ^ 4), data = Wage)

coef(summary(fita))
coef(summary(fitb))
coef(summary(fitc))

# 2. Use the function "poly" inside lm() to answer 1 using a) raw = F; b) raw = T. Plot the predictors in each case. 

fit = lm(wage ~ poly(age, 4), data = Wage)
fit2 = lm(wage ~ poly(age, 4, raw = T), data = Wage) 

coef(summary(fit))
coef(summary(fit2))

predictors = model.matrix(fit)
predictors2 = model.matrix(fit2)

index = order(age)
par(mfrow=c(1,2));
plot(predictors[index,1],ylim=range(predictors),type='l', ylab = "orthogonal basis", xlab = "age");
for(i in 2:5) lines(predictors[index,i],col=i)
plot(predictors2[index,1],ylim=range(predictors2),type='l', ylab = "monomials as regressors", xlab = "age");
for(i in 2:5) lines(predictors2[index,i],col=i)

# 2.1 Are the models the same? 

# => the models are the same if raw = T inside poly()
                                        
# 2.2. Are the predictions the same? Compute the predictions for the grid

# agelims = range(age)
# age.grid = seq(from = agelims[1], to = agelims[2])

preds = predict(fit, newdata = list(age = age.grid), se = TRUE)
preds2 = predict(fit2, newdata = list(age = age.grid), se = TRUE)

max(abs(preds$fit - preds2$fit))
plot(abs(preds$fit - preds2$fit),ylim=c(-0.1,0.1))

# => the predictions are the same for raw = T or F.  

# 2.3 Plot the predictions for obs age.grid

se.bands = cbind(preds$fit + 2 * preds$se.fit, preds$fit - 2 * preds$se.fit )

plot(age, wage, xlim = agelims, cex =.5, col="darkgrey")
title("Degree-4 Polynomial ", outer = T)
lines(age.grid, preds$fit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty = 3)

# 3. Do we really need a 4th degree polynomial in the model? Use the anova function to check. 

fit.1 = lm(wage ~ age,data = Wage)
fit.2 = lm(wage ~ poly(age, 2), data = Wage)
fit.3 = lm(wage ~ poly(age, 3), data = Wage)
fit.4 = lm(wage ~ poly(age, 4), data = Wage) 
fit.5 = lm(wage ~ poly(age, 5), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5)

# => degree 4 is at the border. Degree 3 or 4 is ok. 

# 4. Consider the response

# y = 1, if wage > 250 
# y = 0, if wage <= 250

# and poly(age, 4) as the predictors. Make a plot of the predictions in the grid age.grid

fit = glm(I(wage>250) ~ poly(age,4), data = Wage, family = binomial)
preds = predict(fit, newdata = list(age = age.grid), se = T)
se.bands.logit = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

pfit = exp(preds$fit ) / (1 + exp(preds$fit))
se.bands = exp(se.bands.logit) / (1+exp(se.bands.logit))

plot(age, I(wage>250), xlim = agelims, type = "n", ylim = c(0,.2))
lines(age.grid, pfit, lwd = 2, col="blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty = 3)

# 5. Fit wages using a discretized version of age, by using cut(age,4) 

fit = lm(wage ~ cut(age, 4), data = Wage)
coef(summary(fit))

plot(cut(age, 4),fit$fitted)
#17.9, 33.5, 49, 64.5, 80.1

# 6. For the next questions we use the library splines

library(splines)

# 6.1. Fit wage using regression splines  using the b-splines (bs function) for knots located at 25, 40 and 60. Plot the results. Compute the degrees of freedom. 

fit = lm(wage ~ bs(age, knots = c(25,40,60)), data = Wage)
pred = predict(fit, newdata = list(age = age.grid), se = T)
plot(age, wage, col = "gray")
lines(age.grid, pred$fit, lwd=2)
lines(age.grid, pred$fit + 2 * pred$se, lty="dashed")
lines(age.grid, pred$fit - 2 * pred$se, lty="dashed")

df = dim(bs(age,knots=c(25,40,60))) + 1

# 6.2. Fit wage using natural splines (ns function) for knots located at 25, 40 and 60. 

fit2 = lm(wage ~ ns(age, knots=c(25,40,60)), data = Wage)
pred2 = predict(fit2, newdata = list(age = age.grid), se = T)
plot(age, wage, col = "gray")
lines(age.grid, pred2$fit, lwd=2)
lines(age.grid, pred2$fit + 2 * pred$se, lty="dashed")
lines(age.grid, pred2$fit - 2 * pred$se, lty="dashed")

plot(age, wage, col = "gray")
lines(age.grid, pred$fit, lwd=2,col=1)
lines(age.grid, pred2$fit, lwd=2,col=2)

# 6.3. Fit wage using smoothing splines  (smooth.spline function). compare the case df = 16 with cv = T.

fit = smooth.spline(age, wage, df = 16)
fit2 = smooth.spline(age, wage, cv = TRUE)
fit2$df

plot(age, wage, xlim = agelims, cex=.5, col = "darkgrey")
lines(fit, col="red", lwd=2)
lines(fit2, col="blue", lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)


#7. Fit wage using local regression via loess. Compare the results using span =0.5 and span = 0.2
plot(age, wage, xlim = agelims, cex = .5, col = "darkgrey")
fit  = loess(wage ~ age, span = .2, data = Wage)
fit2 = loess(wage ~ age, span = .5, data = Wage)
lines(age.grid, predict(fit, data.frame(age = age.grid)), col = "red", lwd = 2)
lines(age.grid, predict(fit2, data.frame(age = age.grid)), col="blue", lwd = 2)
legend("topright", legend = c("Span=0.2", "Span=0.5"), col = c("red","blue"), lty = 1, lwd = 2, cex = .8)







