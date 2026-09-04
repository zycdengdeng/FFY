#!/usr/bin/env Rscript
# P3 · 100 树 × 2 骨架,phylolm 逐棵拟合,汇总 beta/lambda 的树间分布
# 用法: Rscript src/stage6_surrogate/pgls_p3_full.R [根目录,默认 .]
suppressMessages({ library(ape); library(phylolm) })
args <- commandArgs(trailingOnly = TRUE)
ROOT <- if (length(args) >= 1) args[1] else "."
BT   <- file.path(ROOT, "data/birdtree")
OUT  <- file.path(ROOT, "outputs/pgls"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

dat <- read.csv(file.path(BT, "pgls_data_matched.csv")); rownames(dat) <- dat$tip
res <- list(); k <- 0
for (bb in c("hackett", "ericson")) {
  trees <- read.tree(file.path(BT, paste0(bb, "_100.nwk")))
  cat(sprintf("[%s] %d 棵\n", bb, length(trees)))
  for (i in seq_along(trees)) {
    tr <- keep.tip(trees[[i]], intersect(trees[[i]]$tip.label, dat$tip))
    d  <- dat[tr$tip.label, ]
    m  <- try(phylolm(u ~ HWI + log_m, data = d, phy = tr, model = "lambda"), silent = TRUE)
    if (inherits(m, "try-error")) { cat(sprintf("  %s#%d 失败\n", bb, i)); next }
    sm <- summary(m)$coefficients
    k <- k + 1
    res[[k]] <- data.frame(backbone = bb, tree = i,
                           beta = sm["HWI","Estimate"], se = sm["HWI","StdErr"],
                           p = sm["HWI","p.value"], lambda = m$optpar,
                           beta_m = sm["log_m","Estimate"], n = Ntip(tr))
    if (i %% 20 == 0) cat(sprintf("  …%d/100\n", i))
  }
}
R <- do.call(rbind, res)
write.csv(R, file.path(OUT, "pgls_p3_results.csv"), row.names = FALSE)
cat("\n================ P3 汇总 ================\n")
for (bb in unique(R$backbone)) {
  S <- R[R$backbone == bb, ]
  cat(sprintf("%s  n树=%d  beta中位 %+.4f  [2.5%%,97.5%%]=[%+.4f,%+.4f]  lambda中位 %.3f  p<0.05 比例 %.0f%%\n",
      bb, nrow(S), median(S$beta), quantile(S$beta,.025), quantile(S$beta,.975),
      median(S$lambda), 100*mean(S$p < .05)))
}
cat("→ outputs/pgls/pgls_p3_results.csv\n")
