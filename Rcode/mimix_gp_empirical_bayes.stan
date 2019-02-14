data {
  int<lower=1> N;
  vector[2] coord[N];
  matrix[N,3] covariates;
  vector[N] y;
  int<lower=1> linear_sigma;
}

transformed data {
  matrix[N, N] cov = cov_exp_quad(coord, sigma, lengthscale) + diag_matrix(rep_vector(sigma_e + 1e-10, N));
  matrix[N, N]L_cov = cholesky_decompose(cov);
  vector[N] mu = rep_vector(alpha,N) + covariates*beta;
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> sigma;
  real<lower=0> sigma_e;
  real<lower=0> alpha;
  vector[3] beta;
}

model {
  // prior models
  beta ~ multi_normal(rep_vector(0,3),diag_matrix(rep_vector(linear_sigma,3)));
  lengthscale ~ student_t(4,0,.1);
  sigma ~ student_t(4,0,1);
  sigma_e ~ normal(0,10);
  alpha ~ normal(0,10);
  
  //target += multi_normal_lpdf(beta | rep_vector(0,4), diag_matrix(rep_vector(linear_sigma,4)));
  //target += student_t_lpdf(lengthscale | 4, 0, 1);
  //target += student_t_lpdf(sigma | 4, 0, 1);
  //target += normal_lpdf(sigma | 0, 1);
  
  // observation model
  y ~ multi_normal_cholesky(mu, L_cov);
}
