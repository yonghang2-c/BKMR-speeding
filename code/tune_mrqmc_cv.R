# tune_mrqmc_cv.R

library(bkmr)
library(glmnet)
library(randtoolbox)
library(cluster)
library(foreach)
library(doParallel)
library(doRNG)

#' Construct a MR-QMC-IFF feature mapping function
#'
#' @param Dg       The dimension of global random samples
#' @param centers  A matrix of cluster centers (Kc × p)
#' @param clu_vec  A vector of training set cluster labels (length n_tr)
#' @param Dloc     The local sample dimension per cluster
#' @param seed     Random seed for reproducibility
#' @return A function mapper(Z, clu) that returns combined global and local features
make_mrqmc_mapper <- function(Dg, centers, clu_vec, Dloc, seed = 0) {
  set.seed(seed)
  p <- ncol(centers)
  
  # Global component: Sobol sequence and transformation
  U_g     <- sobol(n = Dg, dim = p, scrambling = FALSE)
  omega_g <- t(qnorm(U_g))
  b_g     <- runif(Dg, 0, 2*pi)
  
  # Local component: one set per cluster
  Kc      <- nrow(centers)
  omega_l <- vector("list", Kc)
  b_l     <- vector("list", Kc)
  for (k in seq_len(Kc)) {
    set.seed(seed + k)
    U_l          <- sobol(n = Dloc, dim = p, scrambling = FALSE)
    omega_l[[k]] <- t(qnorm(U_l))
    b_l[[k]]     <- runif(Dloc, 0, 2*pi)
  }
  
  # Return the mapping function
  function(Z, clu) {
    # Compute global features
    Zg <- sqrt(2/Dg) * cos(Z %*% omega_g + rep(b_g, each = nrow(Z)))
    
    # Initialize local feature matrix
    Zl <- matrix(0, nrow(Z), Kc * Dloc)
    # Fill in local features for each cluster
    for (k in seq_len(Kc)) {
      idx <- which(clu == k)
      if (length(idx) > 0) {
        Zl[idx, ((k-1)*Dloc + 1):(k*Dloc)] <-
          sqrt(2/Dloc) * cos(
            Z[idx, , drop = FALSE] %*% omega_l[[k]] +
              rep(b_l[[k]], each = length(idx))
          )
      }
    }
    # Combine global and local features
    cbind(Zg, Zl)
  }
}

#' Perform single-layer CV hyperparameter search for MR-QMC-IFF
#'
#' @param z_tr         Training feature matrix (n_tr × p)
#' @param y_tr         Response vector for training (length n_tr)
#' @param z_te         Test feature matrix (n_te × p)
#' @param foldid       Fold ID vector for cv.glmnet (length n_tr)
#' @param D_tot_cands  Candidate values for total feature dimension; default: 50 evenly spaced between 0.1·n_tr and 0.9·n_tr
#' @param Dg_ratios    Ratios for the global dimension; default: sequence from 0.10 to 0.50 of length 9
#' @param Kc_cands     Candidate numbers of clusters; default: c(3, 5, 10)
#' @param ncores       Number of cores for parallel processing; default: min(detectCores()-1, 4)
#' @return A list with:
#'   - results: data.frame of (Dg, Kc, Dloc, D_tot, lambda_min, MSE, Time)
#'   - best: best parameter set including cluster labels for train and test
#'   - mapper: the best feature mapping function (call mapper(Z, clu))
#'
#' @examples
#' res <- tune_mrqmc_cv(z_tr, y_tr, z_te, foldid)
#' best <- res$best
#' mapper <- res$mapper
#' Z_tr_mrqmc <- mapper(z_tr, best$clu_tr[[1]])
#' Z_te_mrqmc <- mapper(z_te, best$clu_te[[1]])
tune_mrqmc_cv <- function(z_tr, y_tr, z_te, foldid,
                          D_tot_cands = floor(seq(from = 0.1 * length(y_tr),
                                                  to   = 0.9 * length(y_tr),
                                                  length.out = 50)),
                          Dg_ratios   = seq(0.10, 0.50, length.out = 9),
                          Kc_cands    = c(3, 5, 10),
                          ncores      = min(detectCores() - 1, 4)) {
  require(doParallel)
  require(foreach)
  
  n_tr <- length(y_tr)
  p    <- ncol(z_tr)
  
  # Generate candidate length-scale values using median pairwise distance
  samp_idx  <- sample(seq_len(n_tr), size = min(500, n_tr))
  med_dist  <- median(dist(z_tr[samp_idx, , drop = FALSE]))
  ell_cands <- c(med_dist/2, med_dist, med_dist*2)
  
  # Build an initial grid to determine all unique total dimensions
  rff_grid <- expand.grid(D   = D_tot_cands,
                          ell = ell_cands,
                          KEEP.OUT.ATTRS = FALSE,
                          stringsAsFactors = FALSE)
  all_D    <- sort(unique(rff_grid$D))
  minD     <- all_D[1]
  secondD  <- all_D[25]
  D_range  <- seq(from = minD, to = secondD, by = 50)
  
  # Compute candidate global dimensions from D_range and ratios
  Dg_cands <- sort(unique(as.integer(unlist(
    lapply(D_range, function(Dt) floor(Dg_ratios * Dt)))
  )))
  
  # Set up parallel backend
  cl <- makeCluster(ncores)
  registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() }, add = TRUE)
  
  # Pre-clustering for each Kc candidate
  centers_list <- list(); clu_tr_list <- list(); clu_te_list <- list()
  for (Kc in Kc_cands) {
    km <- kmeans(z_tr, centers = Kc, nstart = 10)
    centers_list[[as.character(Kc)]] <- km$centers
    clu_tr_list[[as.character(Kc)]]  <- km$cluster
    # Assign test observations to nearest cluster center
    clu_te_list[[as.character(Kc)]] <- apply(z_te, 1, function(r)
      which.min(colSums((km$centers - r)^2)))
  }
  
  # Construct the candidate grid ensuring Dloc is a positive integer
  mrqmc_grid <- data.frame(Dg=integer(), Kc=integer(), Dloc=integer(), D_tot=integer())
  for (D_tot in D_tot_cands) {
    for (Dg in Dg_cands) {
      for (Kc in Kc_cands) {
        Dloc_raw <- (D_tot - Dg) / Kc
        if (Dloc_raw >= 1 && abs(Dloc_raw - round(Dloc_raw)) < 1e-8) {
          mrqmc_grid <- rbind(
            mrqmc_grid,
            data.frame(
              Dg    = Dg,
              Kc    = Kc,
              Dloc  = as.integer(round(Dloc_raw)),
              D_tot = D_tot
            )
          )
        }
      }
    }
  }
  print(mrqmc_grid)  # Verify that Dloc values are valid integers
  
  # Export necessary objects to cluster workers
  clusterExport(cl, varlist = c("z_tr","y_tr","foldid",
                                "centers_list","clu_tr_list",
                                "make_mrqmc_mapper","p"),
                envir = environment())
  
  # Parallel cross-validation over all parameter combinations
  mrqmc_res <- foreach(i = seq_len(nrow(mrqmc_grid)), .combine = rbind,
                       .packages = c("glmnet","randtoolbox")) %dopar% {
                         params <- mrqmc_grid[i,]
                         Dg     <- params$Dg; Kc <- params$Kc; Dloc <- params$Dloc; D_tot <- params$D_tot
                         centers<- centers_list[[as.character(Kc)]]
                         clu_tr <- clu_tr_list[[as.character(Kc)]]
                         mapper <- make_mrqmc_mapper(Dg, centers, clu_tr, Dloc, seed = i)
                         Zf     <- mapper(z_tr, clu_tr)
                         t0     <- proc.time()
                         cvf    <- cv.glmnet(Zf, y_tr,
                                             alpha       = 0,
                                             foldid      = foldid,
                                             standardize = TRUE)
                         data.frame(Dg          = Dg,
                                    Kc          = Kc,
                                    Dloc        = Dloc,
                                    D_tot       = D_tot,
                                    MSE         = min(cvf$cvm),
                                    lambda_min  = cvf$lambda.min,
                                    Time        = (proc.time() - t0)["elapsed"],
                                    stringsAsFactors = FALSE)
                       }
  
  # Identify the best combination by minimum MSE
  best_idx    <- which.min(mrqmc_res$MSE)
  best        <- mrqmc_res[best_idx,]
  best_Kc     <- best$Kc
  best_clu_tr <- clu_tr_list[[as.character(best_Kc)]]
  best_clu_te <- clu_te_list[[as.character(best_Kc)]]
  
  # Create mapper function for the best parameters
  best_mapper <- make_mrqmc_mapper(best$Dg,
                                   centers_list[[as.character(best_Kc)]],
                                   best_clu_tr,
                                   best$Dloc,
                                   seed = best_idx)
  
  list(
    results = mrqmc_res,
    best    = cbind(best,
                    clu_tr = list(best_clu_tr),
                    clu_te = list(best_clu_te)),
    mapper  = best_mapper
  )
}

