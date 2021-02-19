
gigg <- function(X, C, Y, grp_idx, method = "fixed", alpha_inits = rep(0, ncol(C)), beta_inits = rep(0, ncol(X)), lambda_sq_inits = rep(1, ncol(X)), gamma_sq_inits = rep(1.0, length(unique(grp_idx))), a = rep(0.5, length(unique(grp_idx))), b = rep(0.5, length(unique(grp_idx))),
                 tau_sq_init = 1, sigma_sq_init = 1, nu_init = 1, n_burn_in = 500, n_samples = 1000, n_thin = 1, stable_const = 1e-07, verbose = TRUE, btrick = FALSE, stable_solve = FALSE){
  
  # Store useful quantities
  grp_size <- as.vector(table(grp_idx))
  grp_size_cs <- cumsum(grp_size)
  
  if(method == "fixed") {
    
    # Fit grouped horseshoe
    gigg <- gigg_fixed_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                     alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                     eta_inits = rep(1.0, length(unique(grp_idx))), p = a, q = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                     nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                     verbose = verbose, btrick = btrick, stable_solve = stable_solve)
    return(gigg)
    
  } else if(method == "mmle") {
    
    # Fit GIGG regression with MMLE
    gigg <- gigg_mmle_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                    alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                    eta_inits = rep(1.0, length(unique(grp_idx))), p_inits = a, q_inits = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                    nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                    verbose = verbose, btrick = btrick, stable_solve = stable_solve)
    return(gigg)
    
  } else {
    
    stop("Method argument invalid. Must choose one of method = 'fixed' or
         method = 'mmle'.")
    
  }
  
}

