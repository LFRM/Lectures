setwd("/Users/francisco/Desktop")

data = read.csv("E0.csv")
data$Date = as.Date(strptime(data$Date,"%d/%m/%Y"))

data = data.frame(Date = data$Date,
                  HomeTeam = data$HomeTeam,
                  AwayTeam = data$AwayTeam,
                  FTHG = data$FTHG,
                  FTAG = data$FTAG,
                  FTR = data$FTR,               
                  HS = data$HS,
                  AS = data$AS, 
                  HST = data$HST,
                  AST = data$AST,       
                  HF = data$HF,
                  AF = data$AF,
                  HC = data$HC,
                  AC = data$AC,
                  HY = data$HY,
                  AY = data$AY,
                  HR = data$HR,
                  AR = data$AR)

data = data[order(data$Date,decreasing=T),] 
data = data[data$FTR!="D",]
#example
((data[data$HomeTeam=="Brighton",])[1,-c(1,2,3,6)]+(data[data$HomeTeam=="Brighton",])[2,-c(1,2,3,6)])/2
((data[data$AwayTeam=="Man City",])[1,-c(1,2,3,6)]+(data[data$AwayTeam=="Man City",])[2,-c(1,2,3,6)])/2

#design matrix
response = (data$FTR)[1:20]
teams = unique(data$HomeTeam)

xmat=matrix(NA,ncol=(ncol(data)-4)*2,20)
for(i in 1:25){
    data = data[-seq(1,i),]
    xmat[i,1:14] = as.numeric(((data[data$HomeTeam==data$HomeTeam[i],])[1,-c(1,2,3,6)]+(data[data$HomeTeam==data$HomeTeam[i],])[2,-c(1,2,3,6)])/2)
    xmat[i,15:28] = as.numeric(((data[data$AwayTeam==data$HomeTeam[i],])[1,-c(1,2,3,6)]+(data[data$AwayTeam==data$AwayTeam[i],])[2,-c(1,2,3,6)])/2)
}

glm.fits = glm(as.factor(response) ~ xmat, family = binomial)

