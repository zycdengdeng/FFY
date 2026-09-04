#!/usr/bin/env Rscript
# P2 · 单棵树跑通 —— 关键闸门:λ≈1 且 β₁ 不显著 → 停,回来重新定位结论
# 用法: Rscript src/stage6_surrogate/pgls_p2_single.R [根目录,默认 .]
suppressMessages({ library(ape); library(phylolm); library(nlme) })
args <- commandArgs(trailingOnly = TRUE)
ROOT <- if (length(args) >= 1) args[1] else "."
BT   <- file.path(ROOT, "data/birdtree")

dat <- read.csv(file.path(BT, "pgls_data_matched.csv"))
rownames(dat) <- dat$tip
cat(sprintf("[数据] %d 种\n", nrow(dat)))

tr <- read.tree(file.path(BT, "hackett_100.nwk"))[[1]]
tr <- keep.tip(tr, intersect(tr$tip.label, dat$tip))
d  <- dat[tr$tip.label, ]
cat(sprintf("[树] Hackett 第 1 棵,剪枝后 %d tip\n", Ntip(tr)))

## --- OLS 基线(等于假设 λ=0) ---
o <- lm(u ~ HWI + log_m, data = d)
cat(sprintf("[OLS ] beta_HWI = %+.4f  (t = %.1f)\n",
            coef(o)["HWI"], summary(o)$coefficients["HWI", 3]))

## --- 主打:phylolm(Pagel's lambda) ---
t0 <- Sys.time()
m <- phylolm(u ~ HWI + log_m, data = d, phy = tr, model = "lambda")
sm <- summary(m)$coefficients
cat(sprintf("[PGLS] beta_HWI = %+.4f ± %.4f  (t = %.1f, p = %.2e)   lambda = %.3f   [%.0fs]\n",
            sm["HWI","Estimate"], sm["HWI","StdErr"], sm["HWI","t.value"],
            sm["HWI","p.value"], m$optpar, as.numeric(Sys.time()-t0, units="secs")))

## --- 交叉验证:nlme::gls + corPagel,抽 2000 种(全量 corPagel 太慢,只为证两工具一致) ---
set.seed(1)
sub  <- sample(tr$tip.label, 2000)
trs  <- keep.tip(tr, sub); ds <- d[trs$tip.label, ]; ds$tip <- rownames(ds)
t0 <- Sys.time()
g  <- try(gls(u ~ HWI + log_m, data = ds,
              correlation = corPagel(0.8, trs, form = ~tip), method = "ML"), silent = TRUE)
if (!inherits(g, "try-error")) {
  lam_g <- attr(g$modelStruct$corStruct, "value")  # 位置因 nlme 版本而异,失败就打印 summary
  cat(sprintf("[corPagel·2000种] beta_HWI = %+.4f   lambda = %.3f   [%.0fs]\n",
              coef(g)["HWI"], as.numeric(coef(g$modelStruct$corStruct, unconstrained=FALSE)),
              as.numeric(Sys.time()-t0, units="secs")))
  ms <- phylolm(u ~ HWI + log_m, data = ds, phy = trs, model = "lambda")
  cat(sprintf("[phylolm·同2000种] beta_HWI = %+.4f   lambda = %.3f   (两工具应一致)\n",
              coef(ms)["HWI"], ms$optpar))
} else cat("[corPagel] 拟合失败(不阻塞,phylolm 为主口径):", attr(g,"condition")$message, "\n")

## --- 闸门判定 ---
b <- sm["HWI","Estimate"]; p <- sm["HWI","p.value"]; lam <- m$optpar
cat("\n================ P2 闸门 ================\n")
if (p < 0.01 && b < 0) {
  cat(sprintf("✓ 通过:控制系统发育后 HWI 效应仍显著为负(beta=%.4f, p=%.1e, lambda=%.2f)\n→ 值得投 P3(100 树 × 2 骨架)。\n", b, p, lam))
} else {
  cat(sprintf("✗ 未过:beta=%.4f p=%.2e lambda=%.2f\n→ 停。按《PGLS_方案》预案改写结论,别跑 P3。\n", b, p, lam))
  quit(status = 1)
}
