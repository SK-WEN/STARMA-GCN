##STARMA model
library(starma)

# Load spdep library to easily create weight matrices
library(spdep)
# Create a 5x5 regular grid which will be our lattice
data=read.csv("test_data.csv")
data=apply(data[1:720,2:(dim(data)[2])],2,as.numeric)
train_data=data[1:(720-12),]
test_data=data[(720-11):720,]
# Create a uniform first order neighbourhood
coordinate_ST=read.csv("test_data(coordinate).csv")
coordinate_ST=coordinate_ST[,2:3]
coordinate_ST=apply(coordinate_ST,2,as.numeric)
plot(coordinate_ST)
knb <- dnearneigh(coordinate_ST, 0, 120,longlat = TRUE)
plot(knb, coordinate_ST)
# Lag the neighbourhood to create other order matrices
knb <- nblag(knb, 2)
klist <- list(order0=diag(9),
order1=nb2mat(knb[[1]]),
order2=nb2mat(knb[[2]]))

write.csv(nb2mat(knb),"adj.csv")

# Simulate a STARMA(2;1) process

data_for_starma=stcenter(train_data)
# Identify the process
stacf(data, klist)
stpacf(data, klist)
# Estimate the process
ar <- matrix(c(1, 1, 1, 1,1,1,1,1), 4, 2)
ma<-matrix(c(0))
model <- starma(data_for_starma, klist, ar, ma)
model
summary(model)
# Diagnose the process
stcor.test(model$residuals, klist, fitdf=4)
stacf(model$residuals, klist)
stpacf(model$residuals, klist)

model$phi
data=as.matrix(data)
dim(as.matrix(data[1,]))
pred<-function(phi,data,adj){
  data=as.matrix(data)
  t=dim(data)[1]
  prediction=0
  for(i in 1:dim(phi)[1]){
    for(j in 1:dim(phi)[2]){
      para=as.matrix(adj[[j]][1,])
      x=as.matrix(data[t-i+1,])
      prediction=prediction+phi[i,j]*t(para)%*%data[t-i+1,]
    }
  }
  return(prediction)
}

prediction=c(0)
for(i in 1:12){
  for_pred=data[1:(720-13+i),]
  prediction=c(prediction,pred(model$phi,for_pred,klist))
}
prediction=prediction[-1]
test_data[,1]
mean(abs(prediction-test_data))
