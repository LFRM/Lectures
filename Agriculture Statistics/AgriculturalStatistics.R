##################
# Some R functions
##################

# some measures of location and dispertion

x = c(3450,3550,3650,3480,3355,3310,3490,3730.3540,3925,3520,3480) #list of values

mean(x) #computes the mean

var(x) #computes the variance

hist(x) #plots a histogram

boxplot(x) #plots a boxplot


# Random variables and simulation

x = c(54,117,72,42,12,3) #list of values

fx = x/sum(x) # computes function f

x=seq(-10,10,0.1) # sequence of values for the random variable

fx=(1/sqrt(2*pi))*exp(-0.5*x*x) # compues f(x) for the rv. x dist as N(0,1).

plot(x,fx,lwd=3,type="l") #plots f(x)

sim = rnorm(10000,mean=0,sd=1) #simulates 10000 values of x, where for the rv. x dist as N(0,1).

plot(sim) #cloud of sitmulated numbers.

plot(sim,col=8) #cloud of simulated numbers.
abline(h=c(mean(sim),-2*sd(sim),2*sd(sim)),lwd=c(2,2,2),lty=c(1,3,3)) #horizontal lines indicating the mean, two sd above it and 2 sd below it.

hist(sim) # histogram of the data

hist(sim); lines(x,fx*5000,lwd=3,type="l"); # histogram of the data and the plot of fx. For comparison we multiply fx*5000.


