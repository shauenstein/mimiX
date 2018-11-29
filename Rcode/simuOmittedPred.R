## simulate an omitted predictor ##

# library(FReibier) # available from github
lapply(c("FReibier", "lattice", "nlme", "mgcv", "spdep", "spind", "ncf"), require, character.only=T)
load(file="mimiX_simu.Rdata")


#### Simulate Data ####
# generate a normal response, in a realistic (but simulated) landscape, with an omitted predictor (by default "x2"):
simData("212", par.response=c(0.8, 1, 0.9, -0.8, -0.6, 0.5, 0.2), interactive=F) # this produces a netCDF data set; note that beta_x1 is higher than default (1 rather than 0.2)!
# read in data:
dats <- extract.ncdf("dataset212.nc")
str(dats)
# first entry are information about the dataset
# second entry is the dataset itself

#### plot data ####
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

correlogLM <- correlog(z=residuals(fmLM), x=dats$data$Lon, y=dats$data$Lat, increment=0.05, resamp=0)
plot(correlogLM)
abline(h=0)
# we can ignore the far right (which is driven by fewer data points)
# --> clear spatial autocorrelation at distances < 0.2

#### GLS ####
# a GLS to attempt somehow embracing SAC:
system.time(fmGLS <- gls(y ~ x4 + I(x4^2) + x3*x4, data=dats$data, correlation=corExp(form=~Lon+Lat))) # takes quite some time: 1 hour on Ubuntu
correlogGLS <- correlog(z=residuals(fmGLS, type="normalized"), x=dats$data$Lon, y=dats$data$Lat, increment=0.05, resamp=0)
# note that type is not the default setting!
plot(correlogGLS)
abline(h=0)
# nicely accommodated; so what does the inferred x1 look like?
# Well, it's exponential fudging kernel imposed on the residuals of the spatial model (the raw residuals, not the normalized ones!).
# How to do that? Don't know ...
x1GLS <- residuals(fmGLS) - residuals(fmGLS, type="normalized")
summary(x1GLS)
levelplot(x1GLS ~ Lon + Lat, data=dats$data)
plot(x1GLS, dats$data$x1, las=1, ylab="truth") # no pattern
cor(x1GLS, dats$data$x1) # -0.13 = no correlation

#### SEVM ####
# spatial eigenvector mapping, see also Bauman et al. (2018 Ecology!)
# build a Gabriel graph from the locations
gabNB <- graph2nb(gabrielneigh(as.matrix(dats$data[, 2:3]), nnmult=4))
gabNB # check that all cells have a link: negative!
plot(gabNB, coords=as.matrix(dats$data[,2:3])) # hm: all seem to be connected here; strange ...

kNB <- knn2nb(knearneigh(as.matrix(dats$data[,2:3]), k=2))
kNB # check!

system.time(fitME <- ME(y ~ x4 + I(x4^2) + x3*x4, data=dats$data, listw=nb2listw(kNB))) # takes a while: 1 hour on ubuntu machine
str(fitME) # it takes 175 vectors to make up for missing predictor!
fmME <- lm(y ~ x4 + I(x4^2) + x3*x4 + fitted(fitME), data=dats$data)
correlogME <- correlog(z=residuals(fmME), x=dats$data$Lon, y=dats$data$Lat, increment=0.05, resamp=0)
# note that type is not the default setting!
plot(correlogME)
abline(h=0) # nice: SAC is gone!
# !!! consider using all MEs and glmnet instead !!! (to save time on forward model selection)

# estimate missing predictor:
mimiX_ME <- fitted(fitME) %*% fmME$coefficients[-c(1:4, 180)]
levelplot(mimiX_ME ~ Lon + Lat, data=dats$data)
cor(mimiX_ME, dats$data$x1) # correlation between reconstruction and x1: 0.792 (surprisingly good, I'd say!)
plot(mimiX_ME, dats$data$x1, las=1, ylab="truth")
abline(0,1)
abline(lm(dats$data$x1 ~ mimiX_ME), col="darkgreen", lwd=2)


#### wavelets ####
system.time(fmWRM <- WRM(y ~ x4 + I(x4^2) + x3*x4, data=dats$data, family="gaussian", coord=as.matrix(round(dats$data[,2:3]*1000)), level=1)) # 21 min
summary(fmWRM)
str(fmWRM)
levelplot(fmWRM$fitted.sm ~ Lon + Lat, data=dats$data) # looks as if there is not much spatial autocorrelation
cor(fmWRM$fitted.sm, dats$data$x1) # 0.063 ...


#### GAM-based trend surface ####
fmGAM <- gam(y ~ x4 + I(x4^2) + x3*x4 + ti(Lon, Lat), data=dats$data)
summary(fmGAM)
newdats <- dats$data
newdats[, 5:11] <- 0
fitGAM <- predict(fmGAM, newdata=newdats)
levelplot(fitGAM ~ Lon + Lat, data=dats$data) # looks as if there is not much spatial autocorrelation
cor(fitGAM, dats$data$x1) # 0.4811 ...
plot(fitGAM, dats$data$x1, las=1, ylab="truth")
abline(lm(dats$data$x1 ~ fitGAM), col="darkgreen", lwd=2)

#### GP for ommitted x1 ####
# GPfit holds 2000 samples of predictions. The variance of mean predictive posteriors
# does not capture the covariance between the data  points (spatial correlation)
# -> we need to conduct a Monte Carlo approximation (2000 samples) on the predictive posterior.
# By using only the mean pred we get a higher correlation (.83) than by using
# any MC sample.
beta_estimates <- read.table("MATLABcode/Beta_estimate.txt", sep = ",", col.names = c("E","Var"))
fitGP <- read.table("MATLABcode/Omitted_pred_mean.txt", sep = ",") # 2500X1 vector
fitGP <- read.table("MATLABcode/Omitted_pred.txt", sep = ",") # 2500X2000 matrix
levelplot(fitGP[,500] ~ Lon + Lat, data=dats$data) # the 500th realization of the predictive posterior
levelplot(x1 ~ Lon + Lat, data=dats$data)
corGP <- 0
for (i in 1:ncol(fitGP)) {
  corGP[i] = cor(fitGP[,i],dats$data$x1)
}
hist(corGP) # correlation falls between .77 and .82 with mean .7936
quantile(corGP, c(.025,.5,.975))

#### coefficient results ####
methods <- c("truth", "lm", "GLS", "SEVM", "WRM", "GAM")
no.of.covariates <- length(strsplit(dats$readme$Response_coefficients, "+", fixed=T)[[1]]) -1
coef.res <- matrix(NA, nrow=length(methods), ncol=no.of.covariates)
rownames(coef.res) <- methods
colnames(coef.res) <- strsplit(dats$readme$Response_coefficients, "+", fixed=T)[[1]][-1]

# now the same for standard errors of coefficients
coef.se.res <- coef.res

### THIS IS HORRIBLE and needs to be automatised! Note that sequence in simData is DIFFERENT from those in coefficients !!! ###
coef.res[1,] <- c(1, 0.9, -0.8, -0.6, 0.5)
coef.res[2,] <- c(0, coef(fmLM)[-1][c(1,2,4,3)]) # 0 for omitted x1
coef.res[3,] <- c(0, coef(fmGLS)[-1][c(1,2,4,3)]) # 0 for omitted x1
coef.res[4,] <- c(0, coef(fmME)[-c(1, 5:179)][c(1,2,4,3)]) # 0 for omitted x1
coef.res[5,] <- c(0, fmWRM$b[2:5][c(1,2,4,3)]) # 0 for omitted x1
coef.res[6,] <- c(0, coef(fmGAM)[2:5][c(1,2,4,3)]) # 0 for omitted x1


coef.se.res[1,] <- NA
coef.se.res[2,] <- c(0, summary(fmLM)$coefficients[-1,2][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[3,] <- c(0, summary(fmGLS)$tTable[-1,2][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[4,] <- c(0, summary(fmME)$coefficients[-c(1, 5:179),2][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[5,] <- c(0, fmWRM$"s.e."[2:5][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[6,] <- c(0, summary(fmGAM)$se[2:5][c(1,2,4,3)]) # 0 for omitted x1


#### map comparison: X vs mimiX #### 
# 


save.image(file="mimiX_simu.Rdata")
