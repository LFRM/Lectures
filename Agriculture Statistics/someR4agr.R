# population
N = 10^3
x = runif(N,0,1)
hist(x)
mu = mean(x)

# sample 
n = 2

s2 = combn(x,2)
x2bar = apply(s2,2,mean)
hist(x2bar)

s3 = combn(x,3)
x3bar = apply(s3,2,mean)
hist(x3bar)

debtdata = c(9430,7535,4078,5604,5179,4416,10676,1627,10112,6567,13627,18719,14661,7159,12195,11032,6525,5239,6195,12584,15415,15917,12591,9743,10324,7061,9071,6817,7917,9691,4353,8137,10544,9467,13659,12595,6245,11346,13021,12806,9719,4972,2200,11356,10746,7117,12744,9465,5742,19263,3603,11448,16804,8279,13479,5649,14044,11298,6845,3467,10493,6191,615,12851,13627,5337,12557,8372,6232,7445)

xbardebt = mean(debtdata)
sxbardebt  = sd(debtdata)

# using app
linf = xbardebt - 1.994 * sxbardebt/sqrt(70)
lsup = xbardebt + 1.994 * sxbardebt/sqrt(70)

#using R directly

linf = xbardebt + qt(0.025, df = 69) * sxbardebt/sqrt(70)
lsup = xbardebt + qt(1 - 0.025, df = 69) * sxbardebt/sqrt(70)

linf = 0.44 + qnorm(0.025,mean=0,sd=1) *sqrt(0.44*(1-0.44)/900)
lsup = 0.44 + qnorm(1-0.025,mean=0,sd=1) *sqrt(0.44*(1-0.44)/900)


# regression data
x = c(2,6,8,8,12,16,20,20,22,26)
y = c(58,105,88,118,117,137,157,169,149,202)

# coefficient estimation
b1 = sum( ( x - mean( x ) ) * (y - mean( y ) ) ) /
     sum( ( x - mean( x ) ) ^ 2)
b0 = mean( y ) - b1 * mean( x )

# regression line
fhat = b0 + b1 * x
xp = 10
yp = b0 + b1 * xp
plot(x, y, col = 8); 
lines(x, fhat, col = 4,lwd = 2);abline(h = yp, v = xp, lty =4)

# coefficient of determination
ssr = sum( (fhat - mean( y ) ) ^ 2 )
sst = sum( (y - mean( y ) ) ^ 2 )
sse = sum( ( y - fhat ) ^ 2 )
r2 = ssr / sst

# relation with correlation
cor(x, y)
( - 1 ) ^ ( b1 < 0) * sqrt( ssr / sst )

# ols in one line
ols = lm(y~x)
summary(ols)

# confidence interval
newx = seq(min(x),max(x),by = 0.05)
conf_interval = predict(ols, newdata = data.frame( x = newx ), 
	interval = "confidence", level = 0.95)

plot(x, y, col = 8 )
abline(ols, col = 2, lwd = 2)
lines(newx, conf_interval[ ,2 ], col = "pink", lty = 2, lwd=3)
lines(newx, conf_interval[ ,3 ], col = "pink", lty = 2, lwd=3)

# Prediction interval
pred_interval = predict(ols, newdata = data.frame( x = newx ), 
                interval = "predict", level = 0.95)

plot(x, y, col = 8 )
abline(ols, col = 2, lwd = 2)
lines(newx, conf_interval[ ,2 ], col = "pink", lty = 2, lwd = 3)
lines(newx, conf_interval[ ,3 ], col = "pink", lty = 2, lwd = 3)
lines(newx, pred_interval[ ,2 ], col = "darkgreen", lty = 2, lwd = 3)
lines(newx, pred_interval[ ,3 ], col = "darkgreen", lty = 2, lwd = 3)


