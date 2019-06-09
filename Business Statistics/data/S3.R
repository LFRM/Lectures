################
#Business Statistics
################

setwd("/Users/francisco/Desktop/Lectures/Business Statistics/data")

hours = read.csv("S3hours.csv")$hours
softdrinks = read.csv("S3softdrinks.csv")$softdrinks
audit = read.csv("S3audit.csv")$audit
canswers = read.csv("S3canswers.csv")$canswers
salary = read.csv("S3salary.csv")$salary
marketing = read.csv("S3marketing.csv"); 
restaurant = read.csv("S3restaurant.csv");

# hours
mean(hours) # mean
median(hours) # median

table(hours); table(hours)[which.max(table(hours))] # mode 
x = rnorm(3); print(x); which.max(x) # what is which.max?

# softdrinks
table(softdrinks)/sum(table(softdrinks))

# audit
haudit=hist(audit)
haudit
haudit=hist(audit,breaks=c(10,15,20,25,30,35))
