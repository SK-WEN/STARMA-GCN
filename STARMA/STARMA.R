##STARMA model
library(starma)

# Load spdep library to easily create weight matrices
library(spdep)
# Create a 5x5 regular grid which will be our lattice

# Create a uniform first order neighbourhood
plot(coordinate)
knb <- dnearneigh(coordinate, 0, 135,longlat = TRUE)
plot(knb, coordinate)
# Lag the neighbourhood to create other order matrices
knb <- nblag(knb, 1)
klist <- list(order0=diag(25),
order1=nb2mat(knb))

# Simulate a STARMA(2;1) process
eps <- matrix(rnorm(200*25), 200, 25)
star <- eps
for (t in 3:200) {
star[t,] <- (.4*klist[[1]] + .25*klist[[2]]) %*% star[t-1,] +
(.25*klist[[1]] ) %*% star[t-2,] +
( - .3*klist[[2]]) %*% eps[t-1,] +
eps[t, ]
}
star <- star[101:200,] # Remove first observations
star <- stcenter(star) # Center and scale the dataset
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