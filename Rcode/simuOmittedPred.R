## simulate an omitted predictor ##

library(FReibier) # available from github
?simData

#### Simulate Data ####
# generate a normal response, in a realistic (but simulated) landscape, with an omitted predictor (by default "x2"):
simData("212", par.response=c(0.8, 1, 0.9, -0.8, -0.6, 0.5, 0.2)) # this produces a netCDF data set; note that beta_x1 is higher than default (1 rather than 0.2)!
# read in data:
dats <- extract.ncdf("dataset212.nc")
str(dats)
# first entry are information about the dataset
# second entry is the dataset itself

#### plot data ####
library(lattice)
levelplot(y ~ Lon + Lat, data=dats$data)
levelplot(x1 ~ Lon + Lat, data=dats$data)

#### Fit a LM ####
# fit a "correct" (i.e. all correct but the omitted variable) model and plot residuals as map:
fmLM <- lm(y ~ x4 + I(x4^2) + x3*x4, data=dats$data)
levelplot(residuals(fmLM) ~ Lon + Lat, data=dats$data) # looks as if there is not much spatial autocorrelation

# check for pattern with omitted x1:
summary(lm(residuals(fmLM) ~ dats$data$x1))
# okay, x1 has an effect (and indeed one very close to the truth: 0.2)

#### check for spatial autocorrelation using corelogramm ####
library(ncf)
correlogLM <- correlog(z=residuals(fmLM), x=dats$data$Lon, y=dats$data$Lat, increment=0.05, resamp=0)
plot(correlogLM)
abline(h=0)
# we can ignore the far right (which is driven by fewer data points)
# --> clear spatial autocorrelation at distances < 0.2

#### GLS ####
# a GLS to attempt somehow embracing SAC:
library(nlme)
system.time(fmGLS <- gls(y ~ x4 + I(x4^2) + x3*x4, data=dats$data, correlation=corExp(form=~Lon+Lat))) # takes quite some time: 1 hour on Ubuntu
correlogGLS <- correlog(z=residuals(fmGLS, type="normalized"), x=dats$data$Lon, y=dats$data$Lat, increment=0.05, resamp=0)
# note that type is not the default setting!
plot(correlogGLS)
abline(h=0)
# nicely accommodated; so what does the inferred x1 look like?
# Well, it's exponential fudging kernel imposed on the residuals of the spatial model (the raw residuals, not the normalized ones!).
# How to do that? Don't know ...


#### SEVM ####
# spatial eigenvector mapping, see also Bauman et al. (2018 Ecology!)
library(spdep)
# build a Gabriel graph from the locations^
gabNB <- graph2nb(gabrielneigh(as.matrix(dats$data[, 2:3]), nnmult=4))
gabNB # check that all cells have a link: negative!
plot(gabNB, coords=as.matrix(dats$data[,2:3])) # hm: all seem to be connected here; strange ...

kNB <- knn2nb(knearneigh(as.matrix(dats$data[,2:3]), k=2))
kNB # check!

system.time(fitME <- ME(y ~ x4 + I(x4^2) + x3*x4, data=dats$data, listw=nb2listw(kNB))) # takes a while
str(fitME) # it takes 175 vectors to make up for missing predictor!
fmME <- lm(y ~ x4 + I(x4^2) + x3*x4 + fitted(fitME), data=dats$data)
correlogME <- correlog(z=residuals(fmME), x=dats$data$Lon, y=dats$data$Lat, increment=0.05, resamp=0)
# note that type is not the default setting!
plot(correlogME)
abline(h=0) # nice: SAC is gone!

# estimate missing predictor:
mimiX_ME <- fitted(fitME) %*% fmME$coefficients[-c(1:4, 180)]
levelplot(mimiX_ME ~ Lon + Lat, data=dats$data)
cor.test(mimiX_ME, dats$data$x1) # correlation between reconstruction and x1: 0.7923 (surprisingly good, I'd say!)
plot(mimiX_ME, dats$data$x1, las=1, ylab="truth")
abline(0,1)
abline(lm(dats$data$x1 ~ mimiX_ME), col="darkgreen", lwd=2)

#### wavelets ####
library(spind)
system.time(fmWRM <- WRM(y ~ x4 + I(x4^2) + x3*x4, data=dats$data, family="gaussian", coord=as.matrix(round(dats$data[,2:3]*1000)), level=1))
str(fmWRM)

save.image(file="mimiX_simu.Rdata")