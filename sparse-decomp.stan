functions {
  vector sqrt_vec(vector x) {
    vector[dims(x)[1]] res;
    for (m in 1:dims(x)[1]){
      res[m] <- sqrt(x[m]);
    }
    return res;
  }
}
data {
    int<lower=1> T;  // # time points to regress on
    int<lower=1> G;  // # genes (p in Lorenz's document)
    int<lower=1> R;  // # target variables (r in Lorenz's document)
    real<lower=0> lambda;    // Rate for Laplace prior
    vector[T]     mu_x[G];         // Mean for x
    matrix[T,T]   Sigma_x[G];      // Covariance for x
    vector[T]     mu_y[R];         // Mean for y
    matrix[T,T]   Sigma_y[R];      // Covariance for y
}
transformed data {
    cholesky_factor_cov[T] L_x[G];   // Cholesky factor for Sigma_x
    cholesky_factor_cov[T] L_y[R];   // Cholesky factor for Sigma_y
    for (g in 1:G) {
        L_x[g] <- cholesky_decompose(Sigma_x[g]);
    }
    for (r in 1:R) {
        L_y[r] <- cholesky_decompose(Sigma_y[r]);
    }
}
parameters {
    matrix[G,T]         X;    // Expression profiles
    matrix[G,R]         beta; // Regression coefficients
    vector<lower=0>[R]  s[G]; // Exponential mixing variables
}
model {
    matrix[R,T]    Y;    // Predicted variables
    Y <- beta' * X;      // Regression
    // Laplace prior for beta
    for (g in 1:G) {
        s[g] ~ exponential(square(lambda)/2);
        beta[g] ~ normal(0, sqrt_vec(s[g]));
    }
    // Normal prior for predictors
    for (g in 1:G) {
        X[g] ~ multi_normal_cholesky(mu_x[g], L_x[g]);
    }
    // Normal prior for targets
    for (r in 1:R) {
        Y[r] ~ multi_normal_cholesky(mu_y[r], L_y[r]);
    }
}
