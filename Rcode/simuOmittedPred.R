## simulate an omitted predictor ##

# library(FReibier) # available from github
lapply(c("FReibier", "lattice", "nlme", "mgcv", "spdep", "spind", "ncf", "brms", "rstan"), require, character.only=T)
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
levelplot(x1 ~ Lon + Lat, data=dats$data[1:200,])

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

#### latent spatial GP ####
## GPstuff ##

# Empirical Bayes #

# Omitted_pred_mean has the mean predictive posterior of the latent GP. However the mean value
# alone does not reveal the variance of the predictive posterior.
# -> We need to conduct a Monte Carlo approximation (2000 samples) on the predictive posterior.
# Omitted_pred holds 2000 samples of predictions.
# The mean pred correlates with x1 more strongly (.83) than any MC sample (.77-.82).

# Matern type of covariance function
GPstuff_beta_mean <- read.table("MATLABcode/Matern_gp_beta_estimate.txt", sep = ",", col.names = c("E","Var"))
fitGP <- read.table("MATLABcode/Matern_gp_omitted_pred_mean.txt", sep = ",") # 2500X1 vector
cor_mean = cor(fitGP,dats$data$x1)

fitGP <- read.table("MATLABcode/Matern_gp_omitted_pred.txt", sep = ",") # 2500X2000 matrix
levelplot(fitGP[,500] ~ Lon + Lat, data=dats$data) # the 500th realization of the predictive posterior
levelplot(x1 ~ Lon + Lat, data=dats$data)
corGP <- 0
for (i in 1:ncol(fitGP)) {
  corGP[i] = cor(fitGP[,i],dats$data$x1)
}
hist(corGP) # correlation falls between .78 and .81 with mean .794
quantile(corGP, c(.025,.5,.975))


# Squared exponential type of covariance function
GPstuff_beta_mean <- read.table("MATLABcode/Sqrd_exp_gp_beta_estimate.txt", sep = ",", col.names = c("E","Var"))
fitGP <- read.table("MATLABcode/Sqrd_exp_gp_omitted_pred_mean.txt", sep = ",") # 2500X1 vector
cor_mean = cor(fitGP,dats$data$x1)

fitGP <- read.table("MATLABcode/Sqrd_exp_gp_omitted_pred.txt", sep = ",") # 2500X2000 matrix
levelplot(fitGP[,500] ~ Lon + Lat, data=dats$data) # the 500th realization of the predictive posterior
levelplot(x1 ~ Lon + Lat, data=dats$data)
corGP <- 0
for (i in 1:ncol(fitGP)) {
  corGP[i] = cor(fitGP[,i],dats$data$x1)
}
hist(corGP) # correlation falls between .77 and .82 with mean .7936
quantile(corGP, c(.025,.5,.975))


### BRMS ###
## Fit a hierarchical Bayesian model with brms

# Full Bayesian
brms_data <- list("y" = dats$data$y, "lon" = dats$data$Lon, "lat" = dats$data$Lat,
                  "x4" = dats$data$x4, "x4_2" = dats$data$x4^2, "x3" = dats$data$x3)

prior_all <- set_prior("normal(0,10)", class = "sigma") + 
  set_prior("normal(0,10)", class = "Intercept") + 
  set_prior("normal(0,10)", class = "b") + 
  set_prior("student_t(4,0,1)", class = "sdgp") + 
  set_prior("student_t(4,0,.1)", class = "lscale")

inits <- list(.2, .2, .5, .1, c(.1,.1,.1))
names(inits) <- c("sigma", "lscale", "sdgp", "Intercept", "b")
inits_chain1 <- list(inits)

fit <- brm(y ~ gp(lon, lat) + x4 + x4_2 + x4:x3, brms_data, prior = prior_all, inits = inits_chain1,  chains = 1, iter=100)

# correlation between prediction (with spatial GP) and missing covariate
predict_ef <- predict(fit, re_formula = y ~ gp(lon, lat))
cor(predict_ef[,4],dats$data$x1) # .84 - nice performance (with a subset (500) of data)

# get model structure as a Stan code
brm_model_stan <- stancode(fit)

# However, brms is awfully slow with the full data set. I don't follow the way, how GP
# is implemented in brms. They use the Laplacian eigenvalues of the covariance matrix
# and take the spectral density of the eigenvalues given the covariance function
# parameters. Then they approximate GP value in data points by multiplying Laplacian
# eigenfunctions with a random sample of the spectral density. I would prefer of using
# STAN, where we can formulate the model in an understandable manner


### STAN ###
options(mc.cores = parallel::detectCores())

coord <- matrix(c(dats$data$Lon, dats$data$Lat), nrow = length(dats$data$Lat), ncol = 2)
covariates <- matrix(c(dats$data$x4, dats$data$x4^2, dats$data$x4*dats$data$x3), nrow = length(dats$data$Lat), ncol = 3)

# prior for the variation of the linear weights
linear_sigma = 10
f_data <- list("y" = dats$data$y, "coord" = coord, "x" = covariates,
               "N" = length(dats$data$y), "linear_sigma" = linear_sigma)

## Empirical Bayes
inits <- list(lengthscale = .2, sigma = .2, sigma_e = .5, alpha = .5, beta = c(.1, .1, .1))

# optimize hyperparameters
gp_opt1 <- stan_model(file='mimix_gp_empirical_bayes.stan')
opt_fit <- optimizing(gp_opt1, data=f_data, init=inits, seed=5838298, hessian=FALSE)

# predict with MAP estimates for hyperparameters
pred_data <- list("y" = dats$data$y, "coord" = coord, "N" = length(dats$data$y),
                  lengthscale = opt_fit$par[1],sigma = opt_fit$par[2],
                  sigma_e = opt_fit$par[3])
                  
pred_opt_fit <- stan(file='mimix_gp_empirical_bayes_pred.stan', data=pred_data, iter=1000, warmup=0,
                     chains=1, seed=5838298, refresh=1, algorithm="Fixed_param")
save.image("gp_fit_stan_EB.RData")

# correlation of x1 and f_predict
# store samples in a matrix
load("gp_fit_stan_EB.RData")
n = length(dats$data$y)
iter_n = pred_opt_fit@sim$iter

p_samples <- data.frame(pred_opt_fit@sim$samples[[1]])
f_pred <- p_samples[,1:(ncol(p_samples)-1)]

# correlation of x1 and each sample of f_predict
corGP <- apply(f_pred, 1, function(x) cor(x,dats$data$x1))
hist(corGP)
mean(corGP) # the mean correlation .61

# correlation of x1 and mean f_predict
pred_opt_fit_sum <- summary(pred_opt_fit)
f_pred_mean <- pred_opt_fit_sum$summary[1:n,1]
cor_mean <- cor(f_pred_mean, dats$data$x1) # the mean correlation .62

# mean and MSE of beta estimates
stan_EB_beta_mean <- opt_fit$par[4:7]



## Full Bayes, takes around 8 hours
inits <- list(.2, .2, .5, .5, c(.1, .1, .1), c(rep(.5,length(dats$data$y))))
names(inits) <- c("lengthscale", "sigma", "sigma_e", "alpha", "beta", "f_predict")
inits_chain1 <- list(inits)

dgp_fit <- stan(file='mimix_gp_full_bayes.stan', data=f_data, iter=1000, warmup=100,
                init = inits_chain1, chains=1, seed=2, refresh=1, algorithm="NUTS")
save.image("gp_fit_stan_FB.RData")


# correlation of x1 and f_predict
# store samples in a matrix
load("gp_fit_stan_FB.RData")
n = length(dats$data$y)
iter_n = dgp_fit@sim$iter

p_samples <- data.frame(dgp_fit@sim$samples[[1]])
f_pred <- p_samples[,8:(ncol(p_samples)-1)]


# correlation of x1 and each sample of f_predict
corGP <- apply(f_pred, 1, function(x) cor(x,dats$data$x1))
hist(corGP)
mean(corGP) # the mean correlation .61


# correlation of x1 and mean f_predict
dgp_fit_sum <- summary(dgp_fit)
f_pred_mean <- dgp_fit_sum$summary[8:(n+7),1]
cor_mean <- cor(f_pred_mean, dats$data$x1) # the mean correlation .62

# mean and MSE of beta estimates
stan_FB_beta_mean <- dgp_fit_sum$summary[4:7,1]
stan_FB_beta_se <- dgp_fit_sum$summary[4:7,2]


#### coefficient results ####
methods <- c("truth", "lm", "GLS", "SEVM", "WRM", "GAM", "GPstuff_EB", "GP_Stan_EB", "GP_Stan_FB")
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
coef.res[7,] <- c(0, GPstuff_beta_mean[c(1,2,4,3),1]) # 0 for omitted x1
coef.res[8,] <- c(0, stan_FB_beta_mean[c(1,2,4,3)]) # 0 for omitted x1
coef.res[9,] <- c(0, stan_EB_beta_mean[c(1,2,4,3)]) # 0 for omitted x1


coef.se.res[1,] <- NA
coef.se.res[2,] <- c(0, summary(fmLM)$coefficients[-1,2][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[3,] <- c(0, summary(fmGLS)$tTable[-1,2][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[4,] <- c(0, summary(fmME)$coefficients[-c(1, 5:179),2][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[5,] <- c(0, fmWRM$"s.e."[2:5][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[6,] <- c(0, summary(fmGAM)$se[2:5][c(1,2,4,3)]) # 0 for omitted x1
coef.se.res[7,] <- c(0, stan_FB_beta_se[c(1,2,4,3)]) # 0 for omitted x1



#### map comparison: X vs mimiX #### 
# 


save.image(file="mimiX_simu.Rdata")
