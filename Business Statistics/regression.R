############
# REGRESSION
############

# SIMULATION

n = 30 # number of points
set.seed(123); # set seed to replicate results
x = runif(n, min = 0, max = 100) # create  independent variable x

eps1 = rnorm(n, mean = 0, sd = 25) # create epsilon dist. iid N(0,1)

beta0 = 5 # set value for b0
beta1 = 3 # set value for b1

f = beta0 + beta1 * x # compute f (fixed)
y1 = f + eps1 # simulate dependent variable y

pdf('datasim.pdf')
plot(x, y1, col = 8); lines(x, f, col = 2,lwd = 2)
dev.off()

# COMPUTATION OF b0 and b1
b1 = sum( ( x - mean( x ) ) * (y1 - mean( y1 ) ) ) / sum( ( x - mean( x ) ) ^ 2)
b0 = mean( y1 ) - b1 * mean( x )
fhat = b0 + b1 * x

xp = 10
yp = b0 + b1 * xp

pdf('datafit1.pdf')
plot(x, y1, col = 8); lines(x, f, col = 2,lwd = 2);
lines(x, fhat, col = 4,lwd = 2);abline(h = yp, v = xp, lty =4)
dev.off()

# COEFFICIENT OF DETERMINATION
ssr = sum( (fhat - mean( y1 ) ) ^ 2 )
sst = sum( (y1 - mean( y1 ) ) ^ 2 )
sse = sum( ( y1 - fhat ) ^ 2 )

r2 = ssr / sst
cor(x, y1)
( - 1 ) ^ ( b1 < 0 ) * sqrt( ssr / sst )


# OTHER SIMULATIONS
set.seed(1231); eps2 = rnorm(n,mean = 0, sd = 25)
set.seed(1232); eps3 = rnorm(n,mean = 0, sd = 25)
set.seed(1233); eps4 = rnorm(n,mean = 0, sd = 25)

y2 = beta0 + beta1 * x + eps2
y3 = beta0 + beta1 * x + eps3
y4 = beta0 + beta1 * x + eps4

fit1c = lm( y1 ~ x )$coefficients
fit2c = lm( y2 ~ x )$coefficients
fit3c = lm( y3 ~ x )$coefficients
fit4c = lm( y4 ~ x )$coefficients

pdf('datafitall.pdf')
par( mfrow = c( 2, 2 ), mar = c( 4, 4, 2, 2 ) )
plot( x, y1, col = 8); abline( a = beta0, b = beta1, lwd = 2, col = 2 );
abline( a = fit1c[1], b = fit1c[2], lwd = 2, col = 4 )
plot( x, y2, col = 8); abline( a = beta0, b = beta1, lwd = 2, col = 2 )
abline( a = fit2c[1], b = fit2c[2], lwd = 2, col = 4 )
plot( x, y3, col = 8); abline( a = beta0, b = beta1, lwd = 2, col = 2 )
abline( a = fit3c[1], b = fit3c[2], lwd = 2, col = 4 )
plot( x, y4, col = 8); abline( a = beta0, b = beta1, lwd = 2, col = 2 )
abline( a = fit4c[1], b = fit4c[2], lwd = 2, col = 4 )
par(mfrow = c( 1 , 1 ) )
dev.off()

# SUMMARIZE lm

summary( lm( y1 ~ x ) )

#CONFIDENCE INTERVAL

lm.out = lm( y1 ~ x )
newx = seq(min(x),max(x),by = 0.05)
conf_interval = predict(lm.out, newdata = data.frame( x = newx ), interval = "confidence", level = 0.95)

pdf("ci.pdf")
plot(x, y1, col = 8 )
abline(lm.out, col = 2, lwd = 2)
lines(newx, conf_interval[ ,2 ], col = "pink", lty = 2, lwd=3)
lines(newx, conf_interval[ ,3 ], col = "pink", lty = 2, lwd=3)
dev.off()

#PREDICTION INTERVAL

lm.out = lm( y1 ~ x )
newx = seq(min(x),max(x),by = 0.05)
conf_interval = predict(lm.out, newdata = data.frame( x = newx ), interval = "confidence", level = 0.95)
pred_interval = predict(lm.out, newdata = data.frame( x = newx ), interval = "predict", level = 0.95)

pdf("cipred.pdf")
plot(x, y1, col = 8 )
abline(lm.out, col = 2, lwd = 2)
lines(newx, conf_interval[ ,2 ], col = "pink", lty = 2, lwd=3)
lines(newx, conf_interval[ ,3 ], col = "pink", lty = 2, lwd=3)
lines(newx, pred_interval[ ,2 ], col = "darkgreen", lty = 2, lwd=3)
lines(newx, pred_interval[ ,3 ], col = "darkgreen", lty = 2, lwd=3)
dev.off()
