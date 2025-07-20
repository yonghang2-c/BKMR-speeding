# install.packages(c(
#   "bkmr","glmnet","randtoolbox","cluster",
#   "foreach","doParallel","doRNG","brms","coda"
# ))
library(bkmr)
library(glmnet)
library(randtoolbox)
library(cluster)
library(foreach)
library(doParallel)
library(doRNG)
library(brms)
library(coda)

# Clear environment (useful for debugging; can be removed in production)
rm(list = ls())

# Set global random seed
set.seed(0)
registerDoRNG(0)

# --- 1. Simulate data & split into train/test (n = 2000) ---
n   <- 2000; p <- 5; q <- 3
z   <- matrix(runif(n * p), n, p)
x   <- matrix(rnorm(n * q), n, q)
h   <- function(Z) rowSums(sin(2*pi*Z)) + apply(Z, 1, prod)
beta <- c(1.0, -1.0, 0.5)
y    <- h(z) + x %*% beta + rnorm(n, sd = 0.1)

train_idx <- sample.int(n, size = floor(0.8 * n))
test_idx  <- setdiff(seq_len(n), train_idx)
z_tr <- z[train_idx, , drop = FALSE]
x_tr <- x[train_idx, , drop = FALSE]
y_tr <- y[train_idx]
z_te <- z[test_idx, , drop = FALSE]
x_te <- x[test_idx, , drop = FALSE]
y_te <- y[test_idx]

# Feature centering
z_tr <- scale(z_tr)
z_te <- scale(z_te,
              center = attr(z_tr, "scaled:center"),
              scale  = attr(z_tr, "scaled:scale"))

# --- 2. Parallel backend & 5-fold CV fold id ---
ncores <- min(detectCores() - 1, 4)
cl     <- makeCluster(ncores)
registerDoParallel(cl)
foldid <- sample(rep(1:5, length.out = length(y_tr)))

# --- 3. Mapping function generators ---
make_rff_mapper <- function(D, ell, seed = 0) {
  set.seed(seed)
  omega <- matrix(rnorm(p * D, sd = 1/ell), p, D)
  b     <- runif(D, 0, 2*pi)
  function(Z) sqrt(2/D) * cos(Z %*% omega + rep(b, each = nrow(Z)))
}
make_mrqmc_mapper <- function(Dg, centers, clu_vec, Dloc, seed = 0) {
  set.seed(seed)
  U_g     <- sobol(n = Dg, dim = p, scrambling = FALSE)
  omega_g <- t(qnorm(U_g)); b_g <- runif(Dg, 0, 2*pi)
  Kc      <- nrow(centers)
  omega_l <- vector("list", Kc); b_l <- vector("list", Kc)
  for (k in seq_len(Kc)) {
    set.seed(seed + k)
    U_l            <- sobol(n = Dloc, dim = p, scrambling = FALSE)
    omega_l[[k]]   <- t(qnorm(U_l))
    b_l[[k]]       <- runif(Dloc, 0, 2*pi)
  }
  function(Z, clu) {
    Zg <- sqrt(2/Dg) * cos(Z %*% omega_g + rep(b_g, each = nrow(Z)))
    Zl <- matrix(0, nrow(Z), Kc * Dloc)
    for (k in seq_len(Kc)) {
      idx <- which(clu == k)
      if (length(idx)) {
        Zl[idx, ((k-1)*Dloc + 1):(k*Dloc)] <-
          sqrt(2/Dloc) * cos(Z[idx, , drop = FALSE] %*% omega_l[[k]] +
                               rep(b_l[[k]], each = length(idx)))
      }
    }
    cbind(Zg, Zl)
  }
}

# --- 4. RFF-only hyperparameter search via single-layer CV (align 'total feature dimension') ---
n_tr <- length(y_tr)
D_tot_cands <- floor(seq(from = 0.1 * n_tr,
                         to   = 0.9 * n_tr,
                         length.out = 50))
samp_idx  <- sample(seq_len(n_tr), size = min(500, n_tr))
med_dist  <- median(dist(z_tr[samp_idx, ]))
ell_cands <- c(med_dist/2, med_dist, med_dist*2)
rff_grid <- expand.grid(D   = D_tot_cands,
                        ell = ell_cands)

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
                     data.frame(D   = D,
                                ell = ell,
                                MSE = min(cvf$cvm),
                                Time= (proc.time() - t0)["elapsed"])
                   }
best_rff <- rff_res[which.min(rff_res$MSE), ]
print(best_rff)

# --- 5. MR-QMC-IFF hyperparameter search via single-layer CV
#      (ensure Dg + Kc*Dloc == D_tot and Dloc > 0) ---
all_D <- sort(unique(rff_grid$D))

# Use only the range between the smallest and the second smallest dimensions, step size = 50
minD    <- all_D[1]
secondD <- all_D[25]
D_tot_cands <- seq(from = minD, to = secondD, by = 50)

# e.g., 0.10,0.15,...,0.50
Dg_ratios <- seq(0.10, 0.50, length.out = 9)
Dg_cands  <- sort(unique(as.integer(unlist(
  lapply(D_tot_cands, function(Dt) floor(Dg_ratios * Dt))
))))
Kc_cands <- c(3, 5, 10)

# 5.1 Pre-clustering
centers_list <- list(); clu_tr_list <- list(); clu_te_list <- list()
for (Kc in Kc_cands) {
  km <- kmeans(z_tr, centers = Kc, nstart = 10)
  centers_list[[as.character(Kc)]] <- km$centers
  clu_tr_list [[as.character(Kc)]] <- km$cluster
  clu_te_list [[as.character(Kc)]] <- apply(z_te, 1, function(r)
    which.min(colSums((km$centers - r)^2)))
}

# 5.2 Construct grid, keep only combinations where Dloc is a positive integer
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
print(mrqmc_grid)  # Confirm all Dloc are positive integers

# 5.3 Parallel hyperparameter search
clusterExport(cl,
              varlist = c("z_tr","y_tr","foldid",
                          "centers_list","clu_tr_list",
                          "make_mrqmc_mapper","p"),
              envir = environment()
)
mrqmc_res <- foreach(i = seq_len(nrow(mrqmc_grid)), .combine = rbind,
                     .packages = c("glmnet","randtoolbox")) %dopar% {
                       Dg    <- mrqmc_grid$Dg[i]
                       Kc    <- mrqmc_grid$Kc[i]
                       Dloc  <- mrqmc_grid$Dloc[i]
                       D_tot <- mrqmc_grid$D_tot[i]
                       centers <- centers_list[[as.character(Kc)]]
                       clu_tr  <- clu_tr_list[[as.character(Kc)]]
                       mapper <- make_mrqmc_mapper(Dg, centers, clu_tr, Dloc, seed = i)
                       Zf     <- mapper(z_tr, clu_tr)
                       t0  <- proc.time()
                       cvf <- cv.glmnet(Zf, y_tr,
                                        alpha       = 0,
                                        foldid      = foldid,
                                        standardize = TRUE)
                       data.frame(
                         Dg    = Dg,
                         Kc    = Kc,
                         Dloc  = Dloc,
                         D_tot = D_tot,
                         MSE   = min(cvf$cvm),
                         Time  = (proc.time() - t0)["elapsed"]
                       )
                     }
best_mrqmc <- mrqmc_res[which.min(mrqmc_res$MSE), ]
print(best_mrqmc)

# Stop parallel
stopCluster(cl)

# --- 6. Final training & test comparison ---
results <- data.frame(
  Method    = character(),
  TrainTime = numeric(),
  TestMSE   = numeric(),
  stringsAsFactors = FALSE
)

# 6.1 Full BKMR
bk_t0    <- proc.time()
fit_bkmr <- kmbayes(y = y_tr, Z = z_tr, X = x_tr,
                    iter = 2000, varsel = FALSE, verbose = FALSE)
bk_time  <- (proc.time() - bk_t0)["elapsed"]
sel_bk   <- seq(1001, 2000, by = 2)
pred_bk  <- predict(fit_bkmr,
                    ptype = "mean",
                    Znew  = z_te,
                    Xnew  = x_te,
                    sel   = sel_bk)
res_bk   <- mean((y_te - pred_bk)^2)
results  <- rbind(results,
                  data.frame(
                    Method    = "Full BKMR",
                    TrainTime = bk_time,
                    TestMSE   = res_bk
                  ))

# 6.2 RFF-only → brms sampling
Z_tr_rff <- make_rff_mapper(best_rff$D, best_rff$ell, seed = 0)(z_tr)
Z_te_rff <- make_rff_mapper(best_rff$D, best_rff$ell, seed = 0)(z_te)
df_rff   <- as.data.frame(Z_tr_rff)
colnames(df_rff) <- paste0("Z", seq_len(ncol(df_rff)))
df_rff$y_tr <- y_tr

rff_t0  <- proc.time()
fit_rff <- brm(
  y_tr ~ .,
  data    = df_rff,
  family  = gaussian(),
  prior   = prior(normal(0,1), class="b") +
    prior(normal(0,1), class="Intercept"),
  iter    = 2000,
  warmup  = 1000,
  chains  = 4,
  cores   = ncores,
  silent  = TRUE
)
rff_time <- (proc.time() - rff_t0)["elapsed"]

df_te_rff    <- as.data.frame(Z_te_rff)
colnames(df_te_rff) <- paste0("Z", seq_len(ncol(df_te_rff)))
pred_rff_draws <- posterior_linpred(fit_rff,
                                    newdata    = df_te_rff,
                                    re_formula = NA)
pred_rff      <- colMeans(pred_rff_draws)
res_rff       <- mean((y_te - pred_rff)^2)

results <- rbind(results,
                 data.frame(
                   Method    = sprintf("RFF-only (D=%d, ell=%.1f)",
                                       best_rff$D, best_rff$ell),
                   TrainTime = rff_time,
                   TestMSE   = res_rff
                 ))

# 6.3 MR-QMC-IFF → brms sampling
clu_te  <- clu_te_list[[as.character(best_mrqmc$Kc)]]
mr_map  <- make_mrqmc_mapper(
  best_mrqmc$Dg,
  centers_list[[as.character(best_mrqmc$Kc)]],
  clu_tr_list[[as.character(best_mrqmc$Kc)]],
  best_mrqmc$Dloc,
  seed = 0
)
Z_tr_mr <- mr_map(z_tr, clu_tr_list[[as.character(best_mrqmc$Kc)]])
Z_te_mr <- mr_map(z_te, clu_te)

df_mr          <- as.data.frame(Z_tr_mr)
colnames(df_mr) <- paste0("M", seq_len(ncol(df_mr)))
df_mr$y_tr     <- y_tr

mr_t0  <- proc.time()
fit_mr <- brm(
  y_tr ~ .,
  data    = df_mr,
  family  = gaussian(),
  prior   = prior(normal(0,1), class="b") +
    prior(normal(0,1), class="Intercept"),
  iter    = 2000,
  warmup  = 1000,
  chains  = 4,
  cores   = ncores,
  silent  = TRUE
)
mr_time <- (proc.time() - mr_t0)["elapsed"]

df_te_mr     <- as.data.frame(Z_te_mr)
colnames(df_te_mr) <- paste0("M", seq_len(ncol(df_te_mr)))
pred_mr_draws <- posterior_linpred(fit_mr,
                                   newdata    = df_te_mr,
                                   re_formula = NA)
pred_mr       <- colMeans(pred_mr_draws)
res_mr        <- mean((y_te - pred_mr)^2)

results <- rbind(results,
                 data.frame(
                   Method    = sprintf("MR-QMC-IFF (Dg=%d, Kc=%d, Dloc=%d)",
                                       best_mrqmc$Dg,
                                       best_mrqmc$Kc,
                                       best_mrqmc$Dloc),
                   TrainTime = mr_time,
                   TestMSE   = res_mr
                 ))

# Print final comparison results
print(results)


# ───────────────────────────────────────────────────────────────────────────────
# Full script: Posterior predictive distribution comparison + Model diagnostics (R-hat, ESS)
# Assumes: fit_bkmr, fit_rff, fit_mr, z_te, x_te, df_te_rff, df_te_mr,
# sel_bk (e.g., seq(1001,2000)), y_te are already in the environment
# ───────────────────────────────────────────────────────────────────────────────

# 0. Load necessary packages
library(bkmr)       # ExtractSamps(), predict.kmbayes()
library(brms)       # posterior_predict()
library(posterior)  # as_draws_df(), as_draws_array(), rhat(), ess_bulk(), ess_tail()
library(ggplot2)    # ggplot2
library(dplyr)      # bind_rows(), sample_n()
library(tidyr)      # pivot_longer()

# 1. Extract posterior parameter samples for diagnostics

# 1.1 Full BKMR
samps_bk    <- ExtractSamps(fit_bkmr, sel = sel_bk)
df_par_bk   <- as.data.frame(samps_bk$beta)
colnames(df_par_bk) <- paste0("beta[", seq_len(ncol(df_par_bk)), "]")
df_par_bk   <- df_par_bk %>%
  mutate(.chain = 1,
         .iteration = sel_bk)
draws_bk_par <- as_draws_array(as_draws_df(df_par_bk))

# 1.2 RFF-only (brms)
draws_rff_par <- as_draws_array(fit_rff)

# 1.3 MR-QMC-IFF (brms)
draws_mr_par  <- as_draws_array(fit_mr)

# 1.4 Compute diagnostics
rhat_bk_par      <- rhat(draws_bk_par)
ess_bulk_bk_par  <- ess_bulk(draws_bk_par)
ess_tail_bk_par  <- ess_tail(draws_bk_par)

rhat_rff_par     <- rhat(draws_rff_par)
ess_bulk_rff_par <- ess_bulk(draws_rff_par)
ess_tail_rff_par <- ess_tail(draws_rff_par)

rhat_mr_par      <- rhat(draws_mr_par)
ess_bulk_mr_par  <- ess_bulk(draws_mr_par)
ess_tail_mr_par  <- ess_tail(draws_mr_par)

diag_bk <- data.frame(
  param    = "rhat_bk_par",
  Rhat     = as.numeric(rhat_bk_par),
  ESS_Bulk = as.numeric(ess_bulk_bk_par),
  ESS_Tail = as.numeric(ess_tail_bk_par)
)
diag_rff <- data.frame(
  param    = "rhat_rff_par",
  Rhat     = as.numeric(rhat_rff_par),
  ESS_Bulk = as.numeric(ess_bulk_rff_par),
  ESS_Tail = as.numeric(ess_tail_rff_par)
)
diag_mr <- data.frame(
  param    = "rhat_mr_par",
  Rhat     = as.numeric(rhat_mr_par),
  ESS_Bulk = as.numeric(ess_bulk_mr_par),
  ESS_Tail = as.numeric(ess_tail_mr_par)
)

cat("\n=== Diagnostics: Full BKMR (first 5 parameters) ===\n")
print(head(diag_bk, 5))
cat("\n=== Diagnostics: RFF-only (first 5 parameters) ===\n")
print(head(diag_rff, 5))
cat("\n=== Diagnostics: MR-QMC-IFF (first 5 parameters) ===\n")
print(head(diag_mr, 5))

# 2. Posterior predictive samples

# 2.1 Full BKMR predictive matrix (draws × n_te)
ppc_bk_mat <- SamplePred(
  fit  = fit_bkmr,
  Znew = z_te,
  Xnew = x_te,
  sel  = sel_bk,
  type = "link"
)
draw_bk    <- sample.int(nrow(ppc_bk_mat), 1)
y_bk_draw  <- ppc_bk_mat[draw_bk, ]

# 2.2 RFF-only posterior predictive samples (matrix: draws × n_te)
ppc_rff_mat <- posterior_predict(fit_rff, newdata = df_te_rff)
draw_rff    <- sample.int(nrow(ppc_rff_mat), 1)
y_rff_draw  <- ppc_rff_mat[draw_rff, ]

# 2.3 MR-QMC-IFF posterior predictive samples (matrix: draws × n_te)
ppc_mr_mat <- posterior_predict(fit_mr, newdata = df_te_mr)
draw_mr    <- sample.int(nrow(ppc_mr_mat), 1)
y_mr_draw  <- ppc_mr_mat[draw_mr, ]

# 2.4 Construct data.frames
df_bk_ppc  <- data.frame(y_pred = y_bk_draw,  model = "Full_BKMR")
df_rff_ppc <- data.frame(y_pred = y_rff_draw, model = "RFF_only")
df_mr_ppc  <- data.frame(y_pred = y_mr_draw,  model = "MR_QMC_IFF")
df_obs     <- data.frame(y_pred = y_te,       model = "Observed")

# 2.5 Combine (no downsampling needed; lengths match observed)
df_plot <- bind_rows(df_bk_ppc, df_rff_ppc, df_mr_ppc, df_obs)

# 2.6 Plot density overlay
p <- ggplot(df_plot, aes(x = y_pred, color = model, linetype = model)) +
  geom_density(size = 1) +
  labs(
    x     = "Value",
    y     = "Density",
    title = "Posterior Predictive (one draw) vs Observed"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(p)

# Save the final plot as a PNG file
ggsave("posterior_predictive_density.png",
       plot = p,
       width = 8, height = 6, dpi = 300)







