##STARMA model
library(starma)

# Load spdep library to easily create weight matrices
library(spdep)
# Create a 5x5 regular grid which will be our lattice

# Create a uniform first order neighbourhood
coordinate_ST=read.csv("test_data(coordinate).csv")
coordinate_ST=coordinate_ST[,2:3]
coordinate_ST=apply(coordinate_ST,2,as.numeric)
plot(coordinate_ST)
knb <- dnearneigh(coordinate_ST, 0, 120,longlat = TRUE)
plot(knb, coordinate_ST)
# Lag the neighbourhood to create other order matrices
# knb <- nblag(knb, 1)
klist <- list(order0=diag(25),
order1=nb2mat(knb))

write.csv(nb2mat(knb),"adj.csv")

# Simulate a STARMA(2;1) process

data_for_starma=stcenter(data)
# Identify the process
stacf(data, klist)
stpacf(data, klist)
# Estimate the process
ar <- matrix(c(1, 1, 1, 0), 2, 2)

ma <- matrix(c(0, 1), 1, 2)
model <- starma(star, klist, ar, ma)
model
summary(model)
# Diagnose the process
stcor.test(model$residuals, klist, fitdf=4)
stacf(model$residuals, klist)
stpacf(model$residuals, klist)