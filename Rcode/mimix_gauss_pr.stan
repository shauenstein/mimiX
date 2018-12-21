data {
  int<lower=1> N;
  matrix[N, 2] coord;
  matrix[N, 2] covariates;
  vector[N] y;
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> sigma;
  real beta1;
  real beta2;
  real beta3;
}


model {
  matrix[N, N] cov;
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      cov[i, j] = sigma * exp(-.5* l^-2 *dot_self(x[i] - x[j]));   # squared exponential
      #Sigma[i, j] = s2*exp(-inv_l*pow(dot_self(x[i] - x[j]),0.5) ) ;     # exponential
      cov[j, i] = cov[i, j];
    }
  }
  // diagonal elements
  for (k in 1:N)
    Sigma[k, k] = s2 + s2_epsilon;
  
  y ~ normal(alpha + beta1 * x4 + beta2 * x4_2 + beta3 * x4_x3, sigma);
}

