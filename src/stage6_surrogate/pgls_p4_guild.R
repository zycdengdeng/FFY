#!/usr/bin/env Rscript
# E-R1d · 分组标度的系统发育校正:三模型对照(与非 PGLS 版同构)。
#   M0: logL ~ logm                    (单一截距+单一斜率)
#   M1: logL ~ logm + guild            (分组截距,共同斜率)
#   M2: logL ~ logm * guild            (分组截距+分组斜率)
# 每棵树:phylolm(lambda ML),记 AIC 三个 + lambda + M2 各组斜率。
# 用法: Rscript src/stage6_surrogate/pgls_p4_guild.R <ntree_per_backbone>
suppressMessages({library(ape); library(phylolm)})
args <- commandArgs(TRUE)
NT <- ifelse(length(args) >= 1, as.integer(args[1]), 25)

dat <- read.csv("data/birdtree/pgls_guild_data.csv", stringsAsFactors = FALSE)
dat$guild <- relevel(factor(dat$guild), ref = "ground")   # 基准组=样本最大的地面组
rownames(dat) <- dat$tip
dir.create("outputs/pgls", showWarnings = FALSE, recursive = TRUE)
out <- file("outputs/pgls/pgls_p4_results.csv", "w")
guilds <- levels(dat$guild)
hdr <- c("backbone","tree","n","lambda2","aic0","aic1","aic2",
         paste0("b_", guilds), paste0("se_", guilds))
writeLines(paste(hdr, collapse=","), out)

for (bk in c("hackett", "ericson")) {
  trees <- read.tree(sprintf("data/birdtree/%s_100.nwk", bk))
  for (i in seq_len(min(NT, length(trees)))) {
    tr <- trees[[i]]
    keep <- intersect(tr$tip.label, dat$tip)
    trp <- drop.tip(tr, setdiff(tr$tip.label, keep))
    d <- dat[trp$tip.label, ]
    f0 <- try(phylolm(logL ~ logm, d, trp, model="lambda"), silent=TRUE)
    f1 <- try(phylolm(logL ~ logm + guild, d, trp, model="lambda"), silent=TRUE)
    f2 <- try(phylolm(logL ~ logm * guild, d, trp, model="lambda"), silent=TRUE)
    if (inherits(f2, "try-error")) { cat(bk, i, "FAIL\n"); next }
    co <- summary(f2)$coefficients
    bg <- co["logm", 1]; sg <- co["logm", 2]           # 基准组(ground)斜率
    bs <- c(); ss <- c()
    for (g in guilds) {
      if (g == "ground") { bs <- c(bs, bg); ss <- c(ss, sg); next }
      k <- paste0("logm:guild", g)
      bs <- c(bs, bg + co[k, 1]); ss <- c(ss, sqrt(sg^2 + co[k, 2]^2))  # 保守合成
    }
    row <- c(bk, i, nrow(d), sprintf("%.4f", f2$optpar),
             sprintf("%.2f", AIC(f0)), sprintf("%.2f", AIC(f1)), sprintf("%.2f", AIC(f2)),
             sprintf("%.4f", bs), sprintf("%.4f", ss))
    writeLines(paste(row, collapse=","), out); flush(out)
    cat(sprintf("[%s %d] n=%d λ=%.3f AIC0-2: %.0f/%.0f/%.0f b(watbelow)=%.3f b(canopy)=%.3f\n",
        bk, i, nrow(d), f2$optpar, AIC(f0), AIC(f1), AIC(f2),
        bs[which(guilds=="watbelowsurf")], bs[which(guilds=="canopy")]))
  }
}
close(out)
cat("完成 → outputs/pgls/pgls_p4_results.csv\n")
