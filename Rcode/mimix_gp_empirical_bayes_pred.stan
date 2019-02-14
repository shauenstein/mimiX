functions {
  vector gp_pred_rng(vector[] coord, vector y, real sigma, real lengthscale, real sigma_e) {
    int N = rows(y);
    vector[N] f2;
    
    {
      matrix[N, N] cov = cov_exp_quad(coord, sigma, lengthscale) + diag_matrix(rep_vector(1e-10, N));
      matrix[N, N] cov_obs = cov + diag_matrix(rep_vector(square(sigma_e), N));
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
  vector[N] y;
  real<lower=0> lengthscale;
  real<lower=0> sigma;
  real<lower=0> sigma_e;
}

parameters {}
model {}

generated quantities {
  vector[N] f_predict = gp_pred_rng(coord, y, sigma, lengthscale, sigma_e);
}
