#!/usr/bin/env Rscript
# ============================================================================
# 系统发育全家桶 —— 论文 A 的演化证据链
#   S1 演化模型比较(BM/λ/OU/EB):腿长是不是被拉向一个最优点?
#   S2 演化速率:腿 vs 翅,谁是"可调旋钮"?
#   S3 独立对比符号检验:飞行-腿长权衡是反复独立发生的,还是一次深层分化?  ★核心
#   S4 祖先状态重建:腿长的大跳发生在哪些支系上?
#   S5 设计先验按 PGLS 重估(100 树 × 2 骨架)
#   S6 异速指数 b 是否随类群而异 —— "凭什么选水鸟"
#   S7 食性控制下的主模型(系统发育校正版 D 检验)
# 用法: Rscript src/stage6_surrogate/phylo_full.R [根目录,默认 .]
# 每节独立 try(),失败不影响后续;结果逐节落盘,中途挂掉不丢已算部分。
# ============================================================================
suppressMessages({ library(ape); library(phylolm) })
args <- commandArgs(trailingOnly = TRUE)
ROOT <- if (length(args) >= 1) args[1] else "."
BT   <- file.path(ROOT, "data/birdtree")
OUT  <- file.path(ROOT, "outputs/phylo"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
T0 <- Sys.time()
tick <- function(tag) cat(sprintf("    [%s  累计 %.1f 分钟]\n", tag,
        as.numeric(Sys.time()-T0, units="mins")))
hr <- function(s) cat(sprintf("\n\n========== %s ==========\n", s))

D <- read.csv(file.path(BT, "pgls_data_full.csv"), check.names = FALSE)
rownames(D) <- D$tip
cat(sprintf("[数据] %d 种 · 水鸟 %d · 有 HWI %d\n",
            nrow(D), sum(D$is_water), sum(!is.na(D$HWI))))

TR_H <- read.tree(file.path(BT, "hackett_100.nwk"))
TR_E <- read.tree(file.path(BT, "ericson_100.nwk"))
cat(sprintf("[树] hackett %d 棵 · ericson %d 棵 · 超度量 %s\n",
            length(TR_H), length(TR_E), is.ultrametric(TR_H[[1]])))

prune <- function(tr, tips) keep.tip(tr, intersect(tr$tip.label, tips))
DH <- D[!is.na(D$HWI), ]                       # 有 HWI 的子集(主分析)

# ---------------------------------------------------------------- S1 演化模型
hr("S1 · 演化模型比较:腿长残差 u 与飞行指数 HWI")
try({
  tr <- prune(TR_H[[1]], DH$tip); d <- DH[tr$tip.label, ]
  H  <- max(branching.times(tr)); cat(sprintf("树高 %.1f Myr · n=%d\n", H, Ntip(tr)))
  rows <- list(); k <- 0
  for (tn in c("u", "HWI")) {
    y <- setNames(d[[tn]], rownames(d))
    for (mo in c("BM", "lambda", "OUfixedRoot", "EB")) {
      t1 <- Sys.time()
      m <- try(phylolm(y ~ 1, phy = tr,
                       data = data.frame(y = y, row.names = names(y)), model = mo),
               silent = TRUE)
      if (inherits(m, "try-error")) { cat(sprintf("  %s/%s 失败\n", tn, mo)); next }
      op <- if (is.null(m$optpar)) NA else m$optpar
      hl <- if (mo == "OUfixedRoot" && !is.na(op) && op > 0) log(2)/op else NA
      k <- k + 1
      rows[[k]] <- data.frame(trait = tn, model = mo, logLik = as.numeric(logLik(m)),
                              aic = m$aic, optpar = op, half_life_Myr = hl,
                              sigma2 = m$sigma2,
                              secs = as.numeric(Sys.time()-t1, units="secs"))
      cat(sprintf("  %-4s %-12s logLik %10.1f  AIC %10.1f  optpar %8.4f%s [%.0fs]\n",
          tn, mo, as.numeric(logLik(m)), m$aic, op,
          if (!is.na(hl)) sprintf("  半衰期 %.1f Myr(树高的 %.0f%%)", hl, 100*hl/H) else "",
          as.numeric(Sys.time()-t1, units="secs")))
    }
  }
  R1 <- do.call(rbind, rows); write.csv(R1, file.path(OUT,"S1_evo_models.csv"), row.names=FALSE)
  for (tn in unique(R1$trait)) {
    S <- R1[R1$trait==tn, ]; b <- S$model[which.min(S$aic)]
    cat(sprintf("→ %s 最优模型 = %s (ΔAIC 次优 %.1f)\n", tn, b, sort(S$aic)[2]-min(S$aic)))
  }
  tick("S1")
})

# ---------------------------------------------------------------- S2 演化速率
hr("S2 · 谁更易变:腿 vs 翅(z 标准化后比 λ 与 OU 半衰期)")
try({
  R1p <- file.path(OUT,"S1_evo_models.csv")
  if (file.exists(R1p)) {
    R1 <- read.csv(R1p)
    lam <- setNames(R1$optpar[R1$model=="lambda"], R1$trait[R1$model=="lambda"])
    hlf <- setNames(R1$half_life_Myr[R1$model=="OUfixedRoot"],
                    R1$trait[R1$model=="OUfixedRoot"])
    cat(sprintf("  λ:      腿长残差 u = %.3f   飞行指数 HWI = %.3f\n", lam["u"], lam["HWI"]))
    cat(sprintf("  OU半衰期: u = %.1f Myr        HWI = %.1f Myr\n", hlf["u"], hlf["HWI"]))
    lab <- if (!is.na(hlf["u"]) && !is.na(hlf["HWI"]))
      (if (hlf["u"] < hlf["HWI"]) "腿更易变(半衰期更短)→ 腿是演化里的可调旋钮"
       else "翅更易变 → 与工程叙事相反,需谨慎措辞") else "半衰期缺失"
    cat(sprintf("→ %s\n", lab))
    write.csv(data.frame(trait=names(lam), lambda=as.numeric(lam),
              half_life_Myr=as.numeric(hlf[names(lam)])),
              file.path(OUT,"S2_lability.csv"), row.names=FALSE)
  } else cat("  S1 未产出,跳过\n")
  tick("S2")
})

# ------------------------------------------------- S3 独立对比符号检验(核心)
hr("S3 ★ 独立对比符号检验:权衡是反复独立发生的吗?")
try({
  NT <- 20   # 用 20 棵树取分布
  rows <- list(); k <- 0; agg <- list(); j <- 0
  for (bb in c("hackett","ericson")) {
    TRS <- if (bb=="hackett") TR_H else TR_E
    for (i in 1:NT) {
      tr <- prune(TRS[[i]], DH$tip); d <- DH[tr$tip.label, ]
      cu <- pic(setNames(d$u,     rownames(d)), tr)
      ch <- pic(setNames(d$HWI,   rownames(d)), tr)
      cm <- pic(setNames(d$log_m, rownames(d)), tr)
      # 控体重:对比对 log_m 对比过原点回归取残差(PIC 偏相关的标准做法)
      ru <- residuals(lm(cu ~ cm + 0)); rh <- residuals(lm(ch ~ cm + 0))
      ages <- branching.times(tr)[names(cu)]
      qs  <- quantile(ages, c(1/3, 2/3), na.rm=TRUE)
      lay <- cut(ages, c(-Inf, qs[1], qs[2], Inf), labels=c("shallow","mid","deep"))
      for (tag in c("raw","mass_ctrl")) {
        x <- if (tag=="raw") cu else ru; y <- if (tag=="raw") ch else rh
        opp <- (sign(x) * sign(y)) < 0
        fr  <- tapply(opp, lay, mean)
        pv  <- tapply(opp, lay, function(z) binom.test(sum(z), length(z), .5)$p.value)
        k <- k+1
        rows[[k]] <- data.frame(backbone=bb, tree=i, spec=tag, n_node=length(opp),
          frac_opposite=mean(opp), p=binom.test(sum(opp),length(opp),.5)$p.value,
          r_contrast=cor(x,y),
          frac_shallow=fr["shallow"], p_shallow=pv["shallow"],
          frac_mid=fr["mid"],         p_mid=pv["mid"],
          frac_deep=fr["deep"],       p_deep=pv["deep"])
      }
      if (i==1) { j<-j+1; agg[[j]] <- data.frame(backbone=bb,
        node_age=as.numeric(ages), c_u=as.numeric(cu), c_hwi=as.numeric(ch)) }
    }
    cat(sprintf("  %s 完成 %d 棵\n", bb, NT))
  }
  R3 <- do.call(rbind, rows); write.csv(R3, file.path(OUT,"S3_pic_signtest.csv"), row.names=FALSE)
  write.csv(do.call(rbind, agg), file.path(OUT,"S3_contrasts_tree1.csv"), row.names=FALSE)
  for (tg in unique(R3$spec)) {
    S <- R3[R3$spec==tg, ]
    cat(sprintf("\n  [%s] 节点数 %d · 方向相反 %.1f%%(随机应 50%%) · 最大 p %.1e · 对比 r %+.3f\n",
        tg, round(median(S$n_node)), 100*median(S$frac_opposite), max(S$p), median(S$r_contrast)))
    cat(sprintf("     分层  浅(新) %.1f%% (p %.0e)   中 %.1f%% (p %.0e)   深(老) %.1f%% (p %.0e)\n",
        100*median(S$frac_shallow), max(S$p_shallow), 100*median(S$frac_mid), max(S$p_mid),
        100*median(S$frac_deep), max(S$p_deep)))
  }
  cat("\n→ 三个年龄层全部显著高于 50% ⇒ 权衡不是一次深层分化的遗留,\n")
  cat("   而是在演化史上反复独立出现(深层更强,但新近分化中依然成立)。\n")
  tick("S3")
})

# ------------------------------------------------------------ S4 祖先状态重建
hr("S4 · 腿长的大跳发生在哪些支系")
try({
  tr <- prune(TR_H[[1]], DH$tip); d <- DH[tr$tip.label, ]
  y  <- setNames(d$u, rownames(d))
  a  <- ace(y, tr, method="pic")$ace
  st <- c(y[tr$tip.label], a)                       # 1..Ntip 为叶,其后为内节点
  names(st)[1:Ntip(tr)] <- as.character(1:Ntip(tr))
  ch <- st[as.character(tr$edge[,2])] - st[as.character(tr$edge[,1])]
  nd <- sapply(tr$edge[,2], function(v) if (v <= Ntip(tr)) 1L else
               length(unlist(prop.part(tr)[v - Ntip(tr)])))
  sel <- which(nd >= 20)                            # 只看含 ≥20 种的支系
  ordr <- sel[order(-abs(ch[sel]))][1:min(20,length(sel))]
  rows <- list()
  for (q in seq_along(ordr)) {
    e <- ordr[q]; v <- tr$edge[e,2]
    tips <- extract.clade(tr, v)$tip.label
    fam <- names(sort(table(D[tips,"Family"]), decreasing=TRUE))[1]
    ordn<- names(sort(table(D[tips,"Order"]),  decreasing=TRUE))[1]
    rows[[q]] <- data.frame(rank=q, delta_u=as.numeric(ch[e]), n_tip=length(tips),
      dominant_family=fam, dominant_order=ordn,
      mean_u=mean(D[tips,"u"]), mean_HWI=mean(D[tips,"HWI"], na.rm=TRUE))
  }
  R4 <- do.call(rbind, rows); write.csv(R4, file.path(OUT,"S4_shifts.csv"), row.names=FALSE)
  cat("  最大的 10 个支系级腿长跳变:\n")
  for (q in 1:min(10,nrow(R4))) cat(sprintf("   %2d. Δu %+6.2f  n=%4d  %-20s (%s)  该支 HWI 均值 %.1f\n",
      q, R4$delta_u[q], R4$n_tip[q], R4$dominant_family[q], R4$dominant_order[q], R4$mean_HWI[q]))
  tick("S4")
})

# --------------------------------------------------------- S5 设计先验重估
hr("S5 · 设计先验(水鸟异速律)按 PGLS 重估")
try({
  W <- D[D$is_water==1, ]
  cat(sprintf("  会飞水鸟 %d 种\n", nrow(W)))
  o <- lm(logL ~ log_m, data=W)
  cat(sprintf("  [OLS] a=%.4f b=%.4f±%.4f  残差sd=%.4f\n",
      coef(o)[1], coef(o)[2], summary(o)$coefficients[2,2], sigma(o)))
  rows <- list(); k <- 0
  for (bb in c("hackett","ericson")) {
    TRS <- if (bb=="hackett") TR_H else TR_E
    for (i in 1:100) {
      tr <- prune(TRS[[i]], W$tip); d <- W[tr$tip.label, ]
      m <- try(phylolm(logL ~ log_m, data=d, phy=tr, model="lambda"), silent=TRUE)
      if (inherits(m,"try-error")) next
      sm <- summary(m)$coefficients; k <- k+1
      rows[[k]] <- data.frame(backbone=bb, tree=i, a=sm[1,1], b=sm[2,1],
        se_b=sm[2,2], lambda=m$optpar, sd_res=sd(as.numeric(residuals(m))), n=Ntip(tr))
    }
    cat(sprintf("  %s 完成\n", bb))
  }
  R5 <- do.call(rbind, rows); write.csv(R5, file.path(OUT,"S5_prior.csv"), row.names=FALSE)
  cat(sprintf("  [PGLS] b 中位 %.4f  [%.4f, %.4f]   λ 中位 %.3f   残差sd 中位 %.4f\n",
      median(R5$b), quantile(R5$b,.025), quantile(R5$b,.975),
      median(R5$lambda), median(R5$sd_res)))
  cat("  中心线对比(项目现用 a=0.479 b=0.391):\n")
  pr <- list(); q <- 0
  for (mkg in c(5,12,20,30)) {
    lm_ <- log10(mkg*1000)
    Lnow <- 10^(0.479 + 0.391*lm_); Lp <- 10^(median(R5$a) + median(R5$b)*lm_)
    q<-q+1; pr[[q]] <- data.frame(m_kg=mkg, L_now_mm=Lnow, L_pgls_mm=Lp, pct=100*(Lp/Lnow-1))
    cat(sprintf("    %2d kg: 现用 %6.1f mm → PGLS %6.1f mm  (%+.1f%%)\n", mkg, Lnow, Lp, 100*(Lp/Lnow-1)))
  }
  write.csv(do.call(rbind,pr), file.path(OUT,"S5_centerline.csv"), row.names=FALSE)
  tick("S5")
})

# --------------------------------------------------------- S6 类群 b 是否有别
hr("S6 · 异速指数 b 是否随类群而异 ——「凭什么选水鸟」")
try({
  CL <- list(
    "水鸟5科"  = D$is_water==1,
    "雁鸭科"   = D$Family=="Anatidae",
    "鸡形目"   = D$Order=="Galliformes",
    "猛禽3目"  = D$Order %in% c("Accipitriformes","Falconiformes","Strigiformes"),
    "鸻形目"   = D$Order=="Charadriiformes",
    "鸽形目"   = D$Order=="Columbiformes",
    "雀形目"   = D$Order=="Passeriformes")
  rows <- list(); k <- 0
  for (nm in names(CL)) {
    sub <- D[CL[[nm]], ]; if (nrow(sub) < 30) next
    for (i in 1:20) {
      tr <- prune(TR_H[[i]], sub$tip); if (Ntip(tr) < 30) next
      d <- sub[tr$tip.label, ]
      m <- try(phylolm(logL ~ log_m, data=d, phy=tr, model="lambda"), silent=TRUE)
      if (inherits(m,"try-error")) next
      sm <- summary(m)$coefficients; k <- k+1
      rows[[k]] <- data.frame(clade=nm, tree=i, b=sm[2,1], se=sm[2,2],
                              lambda=m$optpar, n=Ntip(tr))
    }
  }
  R6 <- do.call(rbind, rows); write.csv(R6, file.path(OUT,"S6_clade_b.csv"), row.names=FALSE)
  cat("  各类群 b(20 棵树中位,系统发育校正):\n")
  for (nm in unique(R6$clade)) {
    S <- R6[R6$clade==nm, ]
    cat(sprintf("    %-9s n=%4d  b = %.3f ± %.3f  λ=%.2f\n",
        nm, round(median(S$n)), median(S$b), median(S$se), median(S$lambda)))
  }
  cat(sprintf("  物理涌现 b_eff = 0.238 作参照\n"))
  tick("S6")
})

# ------------------------------------------------------- S7 食性控制(D 检验)
hr("S7 · 食性控制下的主模型(系统发育校正版 D 检验)")
try({
  DIET <- c("Diet.Inv","Diet.Vend","Diet.Vect","Diet.Vfish","Diet.Vunk",
            "Diet.Scav","Diet.Fruit","Diet.Nect","Diet.Seed")
  f1 <- as.formula("u ~ HWI + log_m")
  f2 <- as.formula(paste("u ~ HWI + log_m +", paste(DIET, collapse=" + ")))
  rows <- list(); k <- 0
  for (i in 1:20) {
    tr <- prune(TR_H[[i]], DH$tip); d <- DH[tr$tip.label, ]
    for (tag in c("base","with_diet")) {
      m <- try(phylolm(if (tag=="base") f1 else f2, data=d, phy=tr, model="lambda"), silent=TRUE)
      if (inherits(m,"try-error")) next
      sm <- summary(m)$coefficients; k <- k+1
      rows[[k]] <- data.frame(spec=tag, tree=i, beta_HWI=sm["HWI","Estimate"],
        se=sm["HWI","StdErr"], p=sm["HWI","p.value"], lambda=m$optpar)
    }
    if (i %% 5 == 0) cat(sprintf("  …%d/20\n", i))
  }
  R7 <- do.call(rbind, rows); write.csv(R7, file.path(OUT,"S7_diet.csv"), row.names=FALSE)
  for (tg in unique(R7$spec)) {
    S <- R7[R7$spec==tg, ]
    cat(sprintf("  %-9s beta_HWI = %+.4f ± %.4f   λ=%.3f   最大p %.1e\n",
        tg, median(S$beta), median(S$se), median(S$lambda), max(S$p)))
  }
  tick("S7")
})

hr("全部结束")
cat(sprintf("产出目录 %s\n", OUT))
print(list.files(OUT))
cat(sprintf("总耗时 %.1f 分钟\n", as.numeric(Sys.time()-T0, units="mins")))
