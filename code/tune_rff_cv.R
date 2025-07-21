# tune_rff_cv.R

library(glmnet)
library(randtoolbox)
library(foreach)
library(doParallel)
library(parallel)

#' Create a Random Fourier Features (RFF) mapper function
#'
#' This function returns a mapper that transforms input data Z into the RFF feature space.
#'
#' @param D     Integer. Output dimension of the RFF feature mapping.
#' @param ell   Numeric. Kernel length-scale parameter.
#' @param seed  Integer. Random seed for reproducibility.
#' @return A function mapper(Z) that maps a data matrix Z to the RFF feature space (n_samples × D).
#' @examples
#' mapper <- make_rff_mapper(D = 500, ell = 1.0, seed = 42)
#' Z_rff <- mapper(Z_original)

make_rff_mapper <- function(D, ell, seed = 0) {
  set.seed(seed)
  function(Z) {
    p <- ncol(Z)
    omega <- matrix(rnorm(p * D, sd = 1/ell), p, D)
    b     <- runif(D, 0, 2*pi)
    sqrt(2/D) * cos(Z %*% omega + rep(b, each = nrow(Z)))
  }
}

#' Perform hyperparameter cross-validation for RFF + glmnet on training data
#'
#' This function evaluates combinations of RFF output dimensions (D), kernel length-scales (ell),
#' and glmnet's λ via cross-validated MSE, returning the full results and the best combination.
#'
#' @param z_tr         Numeric matrix (n_tr × p). Training feature matrix.
#' @param y_tr         Numeric vector (length n_tr). Training response values.
#' @param foldid       Integer vector (length n_tr). Fold assignments for cv.glmnet (1..K).
#' @param D_tot_cands  Numeric vector. Candidate output dimensions for RFF (e.g., seq(100,1000,100)).
#' @param ell_cands    Numeric vector or NULL. Candidate kernel length-scales; if NULL,
#'                     automatically set to {median/2, median, median*2}.
#' @param ncores       Integer. Number of cores for parallel processing.
#'                     Default is min(detectCores()-1, 4).
#' @return A list with elements:
#'   - results: A data.frame listing each (D, ell) pair with corresponding MSE, best λ, and computation time.
#'   - best:    A one-row data.frame for the combination with minimal MSE (with D, ell, λ_min, MSE).
#' @examples
#' res <- tune_rff_cv(z_tr, y_tr, foldid,
#'                   D_tot_cands = seq(100, 1000, by = 100))
#' res$best
#' mapper <- make_rff_mapper(res$best$D, res$best$ell, seed = 0)
#' Z_tr_rff <- mapper(z_tr)
#' Z_te_rff <- mapper(z_te)

tune_rff_cv <- function(z_tr, y_tr, foldid,
                        D_tot_cands,
                        ell_cands    = NULL,
                        ncores       = min(detectCores() - 1, 4)) {
  # Set up parallel backend
  cl <- makeCluster(ncores)
  registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() }, add = TRUE)
  
  n_tr <- length(y_tr)
  
  # If ell candidates are not provided, compute based on median pairwise distance
  if (is.null(ell_cands)) {
    samp_idx <- sample(seq_len(n_tr), size = min(500, n_tr))
    med_dist <- median(dist(z_tr[samp_idx, , drop = FALSE]))
    ell_cands <- c(med_dist/2, med_dist, med_dist*2)
  }
  
  # Construct the hyperparameter grid
  rff_grid <- expand.grid(D   = D_tot_cands,
                          ell = ell_cands,
                          KEEP.OUT.ATTRS = FALSE,
                          stringsAsFactors = FALSE)
  
  # Perform parallel cross-validation over the grid
  rff_res <- foreach(i = seq_len(nrow(rff_grid)), .combine = rbind,
                     .packages = c("glmnet","randtoolbox")) %dopar% {
                       D   <- rff_grid$D[i]
                       ell <- rff_grid$ell[i]
                       mapper <- make_rff_mapper(D, ell, seed = i)
                       Zf     <- mapper(z_tr)
                       t0     <- proc.time()
                       cvf    <- cv.glmnet(Zf, y_tr,
                                           alpha       = 0,
                                           foldid      = foldid,
                                           standardize = TRUE)
                       lambda_min <- cvf$lambda.min
                       mse_min    <- min(cvf$cvm)
                       data.frame(D          = D,
                                  ell        = ell,
                                  MSE        = mse_min,
                                  lambda_min = lambda_min,
                                  Time       = (proc.time() - t0)["elapsed"],
                                  stringsAsFactors = FALSE)
                     }
  
  # Select the best hyperparameter combination by minimum MSE
  best_rff <- rff_res[which.min(rff_res$MSE), ]
  
  list(results = rff_res,
       best    = best_rff)
}

