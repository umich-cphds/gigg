
<!-- README.md is generated from README.Rmd. Please edit that file -->

# R package `gigg`

# Group Inverse-Gamma Gamma Shrinkage for Sparse Regression with Grouping Structure

[![](https://img.shields.io/badge/devel%20version-0.1.1-blue.svg)](https://github.com/umich-cphds/gigg)
[![](https://img.shields.io/github/languages/code-size/umich-cphds/gigg.svg)](https://github.com/umich-cphds/gigg)

## Overview

This package implements a Gibbs sampler corresponding to a Group
Inverse-Gamma Gamma (GIGG) regression model with adjustment covariates.
Hyperparameters in the GIGG prior specification can either be fixed by
the user or can be estimated via Marginal Maximum Likelihood Estimation.

## Installation

If the devtools package is not yet installed, install it first:

``` r
install.packages('devtools')
```

``` r
# install the package from Github:
devtools::install_github('umich-cphds/gigg') 
```

Once installed, load the package:

``` r
library(gigg)
```

## Examples

GIGG regression Gibbs sampler with fixed hyperparameters:

``` r
?gigg_fixed

X = concentrated$X
C = concentrated$C
Y = concentrated$Y
grp_idx = concentrated$grps
alpha_inits = concentrated$alpha
beta_inits = concentrated$beta

gf = gigg_fixed(X, C, Y, grp_idx, alpha_inits, beta_inits, lambda_sq_inits = rep(1, ncol(X)),
                gamma_sq_inits = rep(1, length(unique(grp_idx))),
                a = rep(0.5, length(unique(grp_idx))),
                b = rep(0.5, length(unique(grp_idx))),
                tau_sq_init = 1, sigma_sq_init = 1, nu_init = 1, n_burn_in = 500,
                n_samples = 1000, n_thin = 1, stable_const = 1e-07, verbose = TRUE,
                btrick = FALSE, stable_solve = FALSE)
```

GIGG regression Gibbs sampler with hyperparameter estimation via
Marginal Maximum Likelihood Estimation:

``` r
?grouped_igg_mmle

X = concentrated$X
C = concentrated$C
Y = concentrated$Y
grp_idx = concentrated$grps
alpha_inits = concentrated$alpha
beta_inits = concentrated$beta

gmmle = grouped_igg_mmle(X, C, Y, grp_idx, alpha_inits, beta_inits, lambda_sq_inits = rep(1, ncol(X)),
                         gamma_sq_inits = rep(1, length(unique(grp_idx))),
                         a = rep(0.5, length(unique(grp_idx))),
                         b = rep(0.5, length(unique(grp_idx))),
                         tau_sq_init = 1, sigma_sq_init = 1, nu_init = 1,
                         n_burn_in = 500, n_samples = 1000, n_thin = 1,
                         stable_const = 1e-07, verbose = TRUE, btrick = FALSE,
                         stable_solve = FALSE)
```

### Current Suggested Citation
