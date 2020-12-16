#' GIGG regression Gibbs sampler with fixed hyperparameters.
#'
#' @param X A (n x p) matrix of covariates that we want to apply GIGG shrinkage on.
#' @param C A (n x k) matrix of covariates that we want to apply no shrinkage on (typically intercept + adjustment covariates).
#' @param Y A (n x 1) column vector of responses.
#' @param grp_idx A (1 x p) row vector indicating which group of the G groups the p covariates in X belong to.
#' @param alpha_inits A (k x 1) column vector containing initial values for the regression coefficients corresponding to C.
#' @param beta_inits A (p x 1) column vector containing initial values for the regression coefficients corresponding to X.
#' @param lambda_sq_inits A (p x 1) column vector containing initial values for the local shrinkage parameters.
#' @param gamma_sq_inits A (G x 1) column vector containing initial values for the group shrinkage parameters.
#' @param a A (G x 1) column vector of shape parameters for the prior on the group shrinkage parameters.
#' @param b A (G x 1) column vector of shape parameters for the prior on the individual shrinkage parameters.
#' @param tau_sq_init Initial value for the global shrinkage parameter (double).
#' @param sigma_sq_init Initial value for the residual variance (double).
#' @param nu_init Initial value for the augmentation variable (double).
#' @param n_burn_in The number of burn-in samples (integer).
#' @param n_samples The number of posterior draws (integer).
#' @param n_thin The thinning interval (integer).
#' @param stable_const Parameter that controls numerical stability of the algorithm (double).
#' @param verbose Boolean value which indicates whether or not to print the progress of the Gibbs sampler.
#' @param btrick Boolean value which indicates whether or not to use the computational trick in Bhattacharya et al. (2016). Only recommended if number of covariates is much larger than the number of observations.
#' @return A list containing the posterior draws of (1) the regression coefficients (alphas and betas) (2) the individual shrinkage parameters (lambda_sqs) (3) the group shrinkage parameters (gamma_sqs) (4) the global shrinkage parameter (tau_sqs) and (5) the residual error variance (sigma_sqs). The list also contains details regarding the dataset (X, C, Y, grp_idx) and Gibbs sampler details (n_burn_in, n_samples, and n_thin).
#' @examples 
#' X = concentrated$X
#' C = concentrated$C
#' Y = concentrated$Y
#' grp_idx = concentrated$grps
#' alpha_inits = concentrated$alpha
#' beta_inits = concentrated$beta
#' 
#' gf = gigg_fixed(X, C, Y, grp_idx, alpha_inits, beta_inits, lambda_sq_inits = rep(1, ncol(X)),
#'                 gamma_sq_inits = rep(1, length(unique(grp_idx))),
#'                 a = rep(0.5, length(unique(grp_idx))),
#'                 b = rep(0.5, length(unique(grp_idx))),
#'                 tau_sq_init = 1, sigma_sq_init = 1, nu_init = 1, n_burn_in = 500,
#'                 n_samples = 1000, n_thin = 1, stable_const = 1e-07, verbose = TRUE,
#'                 btrick = FALSE, stable_solve = FALSE)
gigg_fixed <- function(X, C, Y, grp_idx, alpha_inits = rep(0, ncol(C)), beta_inits = rep(0, ncol(X)), lambda_sq_inits = rep(1, ncol(X)), gamma_sq_inits = rep(1.0, length(unique(grp_idx))), a = rep(0.5, length(unique(grp_idx))), b = rep(0.5, length(unique(grp_idx))),
                        tau_sq_init = 1, sigma_sq_init = 1, nu_init = 1, n_burn_in = 500, n_samples = 1000, n_thin = 1, stable_const = 1e-07, verbose = TRUE, btrick = FALSE, stable_solve = FALSE){
  #Store useful quantites
  grp_size <- as.vector(table(grp_idx))
  grp_size_cs <- cumsum(grp_size)
  
  #Fit grouped horseshoe
  gigg <- gigg_fixed_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                    alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                    eta_inits = rep(1.0, length(unique(grp_idx))), p = a, q = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                    nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                    verbose = verbose, btrick = btrick, stable_solve = stable_solve)
  return(gigg)
}