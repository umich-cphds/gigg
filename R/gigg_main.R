#' GIGG regression
#' 
#' Perform GIGG (Group Inverse-Gamma Gamma Shrinkage) regression.
#' This package implements a Gibbs sampler corresponding to a Group 
#' Inverse-Gamma Gamma (GIGG) regression model with adjustment covariates. 
#' Hyperparameters in the GIGG prior specification can either be fixed by the 
#' user or can be estimated via Marginal Maximum Likelihood Estimation.
#'
#' @param X A (n x p) matrix of covariates that we want to apply GIGG shrinkage on.
#' @param C A (n x k) matrix of covariates that we want to apply no shrinkage on (typically intercept + adjustment covariates).
#' @param Y A length n vector of responses.
#' @param method Either `fixed` for grouped horseshoe or `mmle` for GIGG regression with MMLE.
#' Defaults to method = "fixed".
#' @param grp_idx A length p integer vector indicating which group of the G groups the p covariates in X belong to.
#' The `grp_idx` vector must be a sequence from 1 to p with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.
#' @param alpha_inits A length k vector containing initial values for the regression coefficients corresponding to C.
#' @param beta_inits A length p vector containing initial values for the regression coefficients corresponding to X.
#' @param a A length G vector of shape parameters for the prior on the group shrinkage parameters.
#' @param b A length G vector of shape parameters for the prior on the individual shrinkage parameters.
#' @param sigma_sq_init Initial value for the residual variance (double).
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
#' Y = as.vector(concentrated$Y)
#' grp_idx = concentrated$grps
#' alpha_inits = concentrated$alpha
#' beta_inits = concentrated$beta
#' 
#' gf = gigg(X, C, Y, method = "fixed", grp_idx, alpha_inits, beta_inits,
#'           n_burn_in = 500, n_samples = 1000, n_thin = 1, stable_const = 1e-07, 
#'           verbose = TRUE, btrick = FALSE, stable_solve = FALSE)
#' gf_mmle = gigg(X, C, Y, method = "mmle", grp_idx, alpha_inits, beta_inits,
#'                 n_burn_in = 500, n_samples = 1000, n_thin = 1, 
#'                 stable_const = 1e-07, verbose = TRUE, btrick = FALSE, 
#'                 stable_solve = FALSE)
gigg = function(X, C, Y, method = "fixed", grp_idx, alpha_inits = rep(0, ncol(C)), beta_inits = rep(0, ncol(X)), a = rep(0.5, length(unique(grp_idx))), b = rep(0.5, length(unique(grp_idx))),
                 sigma_sq_init = 1, n_burn_in = 500, n_samples = 1000, n_thin = 1, stable_const = 1e-07, verbose = TRUE, btrick = FALSE, stable_solve = TRUE) {
  
  lambda_sq_inits = rep(1, ncol(X))
  gamma_sq_inits = rep(1.0, length(unique(grp_idx)))
  tau_sq_init = 1
  nu_init = 1
  
  #Store useful quantites
  grp_size <- as.vector(table(grp_idx))
  grp_size_cs <- cumsum(grp_size)
  
  if(!is.matrix(X) | !is.matrix(C)) {
    stop("X and C must be matrices. At least one is of a different type.")
  }
  
  if(!is.vector(Y)) {
    stop("Y must be a vector.")
  }
  
  n = length(Y)
  p = ncol(X)
  k = ncol(C)
  group_ids = unique(grp_idx)
  G = length(group_ids)
  
  if(n != nrow(X) | n != nrow(C)) {
    stop("X and C matrices must have the same number of rows and match the length of the Y vector.")
  }
  
  if(length(grp_idx) != p) {
    stop("The argument grp_idx must have length p, where p in the number of columns in X.")
  }
  
  if(length(alpha_inits) != k) {
    stop("The argument alpha_inits must have length k, where k is the number of columns in C.")
  }
  
  if(length(beta_inits) != p) {
    stop("The argument beta_inits must have length p, where p in the number of columns in X.")
  }
  
  if(length(lambda_sq_inits) != p) {
    stop("The argument lambda_sq_inits must have length p.")
  }
  
  if(length(gamma_sq_inits) != G) {
    stop("The argument gamma_sq_inits must have length G.")
  }
  
  if(length(a) != G) {
    stop("The argument a must have length G.")
  }
  
  if(length(b) != G) {
    stop("The argument b must have length G.")
  }
  
  if(all.equal(grp_idx, sort(grp_idx)) != TRUE) {
    stop("Groups are out of order. Ensure that grp_idx is an ordered sequence
         from 1 to p with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.")
  }
  
  if(any(diff(group_ids) > 1)) {
    stop("Groups skip a number somewhere in grp_idx. Ensure that grp_idx is an ordered sequence
         from 1 to p with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.")
  }
  
  if(method == "fixed") {
    #Fit grouped horseshoe
    gigg <- gigg_fixed_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                     alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                     eta_inits = rep(1.0, length(unique(grp_idx))), p = a, q = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                     nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                     verbose = verbose, btrick = btrick, stable_solve = stable_solve)
    return(gigg)
  } else if(method == "mmle") {
    #Fit GIGG regression with MMLE
    gigg <- gigg_mmle_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                    alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                    eta_inits = rep(1.0, length(unique(grp_idx))), p_inits = a, q_inits = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                    nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                    verbose = verbose, btrick = btrick, stable_solve = stable_solve)
    return(gigg)
  } else {
    stop("Method must be either `fixed` for grouped horseshoe or `mmle` for GIGG 
         regression with MMLE.")
  }
  
}