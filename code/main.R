# main.R

library(bkmr)
library(glmnet)
library(randtoolbox)
library(cluster)
library(foreach)
library(doParallel)
library(doRNG)
library(brms)
library(coda)

source("tune_rff_cv.R")    #  make_rff_mapper, tune_rff_cv()
source("tune_mrqmc_cv.R")  #  make_mrqmc_mapper, tune_mrqmc_cv()

# 1. Test data
set.seed(0); registerDoRNG(0)
n   <- 2000; p <- 5; q <- 3
z   <- matrix(runif(n * p), n, p)
x   <- matrix(rnorm(n * q), n, q)
h   <- function(Z) rowSums(sin(2*pi*Z)) + apply(Z, 1, prod)
beta <- c(1.0, -1.0, 0.5)
y    <- h(z) + x %*% beta + rnorm(n, sd = 0.1)

train_idx <- sample.int(n, size = floor(0.8 * n))
test_idx  <- setdiff(seq_len(n), train_idx)
z_tr <- scale(z[train_idx, , drop = FALSE])
z_te <- scale(z[test_idx,  , drop = FALSE],
              center = attr(z_tr, "scaled:center"),
              scale  = attr(z_tr, "scaled:scale"))
x_tr <- x[train_idx, , drop = FALSE]
x_te <- x[test_idx,  , drop = FALSE]
y_tr <- y[train_idx]
y_te <- y[test_idx]

# 2. CV fold
ncores <- min(detectCores() - 1, 4)
cl     <- makeCluster(ncores); registerDoParallel(cl)
foldid <- sample(rep(1:5, length.out = length(y_tr)))

# 3. RFF-only
res_rff <- tune_rff_cv(
  z_tr, y_tr, foldid,
  D_tot_cands = floor(seq(0.1 * length(y_tr), 0.9 * length(y_tr), length.out = 50))
)
best_rff <- res_rff$best
cat("RFF-only best:\n"); print(best_rff)

# 4. MR-QMC-IFF 
res_mr <- tune_mrqmc_cv(
  z_tr, y_tr, z_te, foldid,
  D_tot_cands = floor(seq(0.1 * length(y_tr), 0.9 * length(y_tr), length.out = 50)),
  Dg_ratios   = seq(0.10, 0.50, length.out = 9),
  Kc_cands    = c(3, 5, 10)
)
best_mr    <- res_mr$best
mapper_mr  <- res_mr$mapper
cat("MR-QMC-IFF best:\n"); print(best_mr)

# 5. Comparison：Full BKMR / RFF-only / MR-QMC-IFF
results <- data.frame(Method=character(), TrainTime=numeric(), TestMSE=numeric(), stringsAsFactors=FALSE)

# 5.1 Full BKMR
t0   <- proc.time()
fit_bk <- kmbayes(y=y_tr, Z=z_tr, X=x_tr,
                  iter=2000, varsel=FALSE, verbose=FALSE)
t_bk <- (proc.time() - t0)["elapsed"]
sel_bk <- seq(1001, 2000, by=2)
pred_bk <- predict(fit_bk, ptype="mean", Znew=z_te, Xnew=x_te, sel=sel_bk)
mse_bk  <- mean((y_te - pred_bk)^2)
results <- rbind(results, data.frame(Method="Full BKMR", TrainTime=t_bk, TestMSE=mse_bk))

# 5.2 RFF-only → brms
mapper_rff <- make_rff_mapper(best_rff$D, best_rff$ell, seed=0)
Ztr_rff <- mapper_rff(z_tr); Zte_rff <- mapper_rff(z_te)
df_tr_rff <- as.data.frame(Ztr_rff); colnames(df_tr_rff) <- paste0("Z", seq_len(ncol(Ztr_rff)))
df_tr_rff$y <- y_tr

t0   <- proc.time()
fit_rff_br <- brm(y ~ ., data=df_tr_rff, family=gaussian(),
                  prior=prior(normal(0,1), class="b") + prior(normal(0,1), class="Intercept"),
                  iter=2000, warmup=1000, chains=4, cores=ncores, silent=TRUE)
t_rff <- (proc.time() - t0)["elapsed"]

df_te_rff <- as.data.frame(Zte_rff); colnames(df_te_rff) <- paste0("Z", seq_len(ncol(Zte_rff)))
pred_draws_rff <- posterior_linpred(fit_rff_br, newdata=df_te_rff, re_formula=NA)
pred_rff      <- colMeans(pred_draws_rff)
mse_rff      <- mean((y_te - pred_rff)^2)
results <- rbind(results,
                 data.frame(Method=sprintf("RFF-only (D=%d,ℓ=%.1f)", best_rff$D, best_rff$ell),
                            TrainTime=t_rff, TestMSE=mse_rff))

# 5.3 MR-QMC-IFF → brms
clu_tr <- best_mr$clu_tr[[1]]; clu_te <- best_mr$clu_te[[1]]
Ztr_mr <- mapper_mr(z_tr, clu_tr); Zte_mr <- mapper_mr(z_te, clu_te)
df_tr_mr <- as.data.frame(Ztr_mr); colnames(df_tr_mr) <- paste0("M", seq_len(ncol(Ztr_mr)))
df_tr_mr$y <- y_tr

t0   <- proc.time()
fit_mr_br <- brm(y ~ ., data=df_tr_mr, family=gaussian(),
                 prior=prior(normal(0,1), class="b") + prior(normal(0,1), class="Intercept"),
                 iter=2000, warmup=1000, chains=4, cores=ncores, silent=TRUE)
t_mr <- (proc.time() - t0)["elapsed"]

df_te_mr <- as.data.frame(Zte_mr); colnames(df_te_mr) <- paste0("M", seq_len(ncol(Zte_mr)))
pred_draws_mr <- posterior_linpred(fit_mr_br, newdata=df_te_mr, re_formula=NA)
pred_mr       <- colMeans(pred_draws_mr)
mse_mr       <- mean((y_te - pred_mr)^2)
results <- rbind(results,
                 data.frame(Method=sprintf("MR-QMC-IFF (Dg=%d,Kc=%d,Dloc=%d)",
                                           best_mr$Dg, best_mr$Kc, best_mr$Dloc),
                            TrainTime=t_mr, TestMSE=mse_mr))

print(results)

stopCluster(cl)

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
