functions {
  vector gp_pred_rng(vector[] coord, vector y, real sigma, real lengthscale, real sigma_e) {
    int N = rows(y);
    vector[N] f2;
    {
      matrix[N, N] cov = cov_exp_quad(coord, sigma, lengthscale) + diag_matrix(rep_vector(1e-10, N));
      matrix[N, N] cov_obs = cov + diag_matrix(rep_vector(sigma_e, N));
      matrix[N, N] L_K_obs = cholesky_decompose(cov_obs);
      vector[N] L_K_div_y = mdivide_left_tri_low(L_K_obs, y);
      vector[N] K_div_y = mdivide_right_tri_low(L_K_div_y', L_K_obs)';
      vector[N] f2_mu = cov * K_div_y;
      matrix[N, N] v_pred = mdivide_left_tri_low(L_K_obs, cov);
      matrix[N, N] f2_cov =   cov - v_pred' * v_pred;
      f2 = multi_normal_rng(f2_mu, f2_cov);
    }
    return f2;
  }
}

data {
  int<lower=1> N;
  vector[2] coord[N];
  matrix[N,3] covariates;
  vector[N] y;
  int<lower=1> linear_sigma;
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> sigma;
  real<lower=0> sigma_e;
  real<lower=0> alpha;
  vector[3] beta;
}


model {
  matrix[N, N] cov = cov_exp_quad(coord, sigma, lengthscale) + diag_matrix(rep_vector(sigma_e + 1e-10, N));
  matrix[N, N]L_cov = cholesky_decompose(cov);
  vector[N] mu = rep_vector(alpha,N) + covariates*beta;

  // prior models
  beta ~ multi_normal(rep_vector(0,3),diag_matrix(rep_vector(sqrt(linear_sigma),3)));
  lengthscale ~ student_t(4,0,sqrt(.1));
  sigma ~ student_t(4,0,1);
  sigma_e ~ normal(0,sqrt(10));
  alpha ~ normal(0,sqrt(10));
  
  //target += multi_normal_lpdf(beta | rep_vector(0,4), diag_matrix(rep_vector(linear_sigma,4)));
  //target += student_t_lpdf(lengthscale | 4, 0, 1);
  //target += student_t_lpdf(sigma | 4, 0, 1);
  //target += normal_lpdf(sigma | 0, 1);
  
  // observation model
  y ~ multi_normal_cholesky(mu, L_cov);
}

generated quantities {
  vector[N] f_predict = gp_pred_rng(coord, y, sigma, lengthscale, sigma_e);
  vector[N] f_predict2 = gp_pred_rng(coord, y, sqrt(.0847), .049, sigma_e);
}
