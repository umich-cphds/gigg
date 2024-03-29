% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gigg_main.R
\name{gigg}
\alias{gigg}
\title{GIGG regression}
\usage{
gigg(
  X,
  C,
  Y,
  method = "mmle",
  grp_idx,
  alpha_inits = rep(0, ncol(C)),
  beta_inits = rep(0, ncol(X)),
  a = rep(0.5, length(unique(grp_idx))),
  b = rep(0.5, length(unique(grp_idx))),
  sigma_sq_init = 1,
  tau_sq_init = 1,
  n_burn_in = 500,
  n_samples = 1000,
  n_thin = 1,
  verbose = TRUE,
  btrick = FALSE,
  stable_solve = TRUE
)
}
\arguments{
\item{X}{A (n x p) matrix of covariates that to apply GIGG shrinkage on.}

\item{C}{A (n x k) matrix of covariates that to apply no shrinkage on (typically intercept + adjustment covariates).}

\item{Y}{A length n vector of responses.}

\item{method}{Either \code{fixed} for GIGG regression with fixed hyperparameters or \code{mmle} for GIGG regression with MMLE.
Defaults to method = "mmle".}

\item{grp_idx}{A length p integer vector indicating which group of the G groups the p covariates in X belong to.
The \code{grp_idx} vector must be a sequence from 1 to G with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.}

\item{alpha_inits}{A length k vector containing initial values for the regression coefficients corresponding to C.}

\item{beta_inits}{A length p vector containing initial values for the regression coefficients corresponding to X.}

\item{a}{A length G vector of shape parameters for the prior on the group shrinkage parameters.
The \code{a} parameter is only used if the user selects \code{method = 'fixed'}. If \code{method = 'mmle'},
then \code{a = rep(1/n, length(unique(grp_idx)))}.}

\item{b}{A length G vector of shape parameters for the prior on the individual shrinkage parameters. If \code{method = 'mmle'},
then the \code{b} is used as an inital value for the MMLE procedure.}

\item{sigma_sq_init}{Initial value for the residual error variance (double).}

\item{tau_sq_init}{Initial value for the global shrinkage parameter (double).}

\item{n_burn_in}{The number of burn-in samples (integer).}

\item{n_samples}{The number of posterior draws (integer).}

\item{n_thin}{The thinning interval (integer).}

\item{verbose}{Boolean value which indicates whether or not to print the progress of the Gibbs sampler.}

\item{btrick}{Boolean value which indicates whether or not to use the computational trick in Bhattacharya et al. (2016). Only recommended if number of covariates is much larger than the number of observations.}

\item{stable_solve}{Boolean value which indicates whether or not to use Cholesky decomposition during the update of the regression coefficients corresponding to X. In our experience, \code{stable_solve = TRUE} is slightly slower, but more stable.}
}
\value{
A list containing
\itemize{
\item{"draws"}{ - A list containing the posterior draws of \cr
(1) the regression coefficients (alphas and betas) \cr
(2) the individual shrinkage parameters (lambda_sqs) \cr
(3) the group shrinkage parameters (gamma_sqs) \cr
(4) the global shrinkage parameter (tau_sqs) and \cr
(5) the residual error variance (sigma_sqs). \cr The list also contains details
regarding the dataset (X, C, Y, grp_idx) and Gibbs sampler details
(n_burn_in, n_samples, and n_thin).}
\item{"beta.hat"}{ - Posterior mean of betas}
\item{"beta.lcl.95"}{ - 95\% credible interval lower bound of betas}
\item{"beta.ucl.95"}{ - 95\% credible interval upper bound of betas}
\item{"alpha.hat"}{ - Posterior mean of alpha}
\item{"alpha.lcl.95"}{ - 95\% credible interval lower bound of alphas}
\item{"alpha.ucl.95"}{ - 95\% credible interval upper bound of alphas}
\item{"sigma_sq.hat"}{ - Posterior mean of sigma squared}
\item{"sigma_sq.lcl.95"}{ - 95\% credible interval lower bound of sigma sq.}
\item{"sigma_sq.ucl.95"}{ - 95\% credible interval upper bound of sigma sq.}
}
}
\description{
Perform GIGG (Group Inverse-Gamma Gamma) regression.
This package implements a Gibbs sampler corresponding to a Group
Inverse-Gamma Gamma (GIGG) regression model with adjustment covariates.
Hyperparameters in the GIGG prior specification can either be fixed by the
user or can be estimated via Marginal Maximum Likelihood Estimation.
}
\examples{
X = concentrated$X
C = concentrated$C
Y = as.vector(concentrated$Y)
grp_idx = concentrated$grps

gf = gigg(X, C, Y, method = "fixed", grp_idx, n_burn_in = 200, 
          n_samples = 500, n_thin = 1, verbose = TRUE, btrick = FALSE, 
          stable_solve = FALSE)
\donttest{
gf_mmle = gigg(X, C, Y, method = "mmle", grp_idx, n_burn_in = 200, 
               n_samples = 500, n_thin = 1, verbose = TRUE, btrick = FALSE, 
               stable_solve = FALSE)
}

}
\references{
Boss, J., Datta, J., Wang, X., Park, S.K., Kang, J., & Mukherjee, B.
(2021). Group Inverse-Gamma Gamma Shrinkage for Sparse Regression with Block-Correlated Predictors.
\href{https://arxiv.org/abs/2102.10670}{arXiv}
}
