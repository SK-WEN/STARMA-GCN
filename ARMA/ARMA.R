library(tseries)
library(forecast)
library(xts)

data=read.csv("test_data.csv")
data=apply(data,2,as.numeric)
data=as.matrix(data)
all_data=data[1:720,2]
train_data=all_data[1:(720-12)]
data_series=ts(train_data,seq(as.POSIXct("2010-09-01"),len=length(train_data),by="hour"))
fit=auto.arima(data_series)


pred=c(0)
for(i in 1:12){
  pred_data=all_data[1:(720-12+i-1)]
  pred_data=ts(pred_data)
  fore=forecast(pred_data,model=fit,h = 1)$mean
  pred=c(pred,fore)
}

pred=pred[-1]
test_data=all_data[(720-11):720]
mean(abs(pred-test_data))

pred-test_data

fit <- auto.arima(WWWusage)
fore=forecast(pred_data,model=fit,h = 12)
plot(forecast(pred_data,model=fit,h = 12))
