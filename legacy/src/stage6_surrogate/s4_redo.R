suppressMessages(library(ape))
U <- "/mnt/user-data/uploads/FFY/FFY"; OUT <- "/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
tr0 <- read.tree(file.path(U,"data/birdtree/hackett_tree1.nwk"))
D <- read.csv(file.path(U,"data/birdtree/pgls_data_full.csv")); rownames(D)<-D$tip
D <- D[!is.na(D$HWI),]
tr <- keep.tip(tr0, intersect(tr0$tip.label,D$tip)); D <- D[tr$tip.label,]
y <- setNames(D$u, rownames(D)); a <- ace(y, tr, method="pic")$ace
st <- c(y[tr$tip.label], a); names(st)[1:Ntip(tr)] <- as.character(1:Ntip(tr))
ch <- st[as.character(tr$edge[,2])] - st[as.character(tr$edge[,1])]
pp <- prop.part(tr)
nd <- sapply(tr$edge[,2], function(v) if (v<=Ntip(tr)) 1L else length(pp[[v-Ntip(tr)]]))
sel <- which(nd>=20); ordr <- sel[order(-abs(ch[sel]))][1:40]
rows <- list()
for (q in seq_along(ordr)) {
  e<-ordr[q]; v<-tr$edge[e,2]; tips<-extract.clade(tr,v)$tip.label
  ft<-sort(table(D[tips,"Family"]),decreasing=TRUE); ot<-sort(table(D[tips,"Order"]),decreasing=TRUE)
  pf<-ft[1]/length(tips); po<-ot[1]/length(tips)
  lvl <- if (pf>=0.7) "family" else if (po>=0.7) "order" else "mixed"
  lab <- if (lvl=="family") names(ft)[1] else if (lvl=="order") names(ot)[1] else
         paste0(names(ot)[1],"+")
  rows[[q]] <- data.frame(rank=q, delta_u=as.numeric(ch[e]), n_tip=length(tips),
    level=lvl, label=lab, purity=round(max(pf,po),2),
    top_family=names(ft)[1], top_order=names(ot)[1],
    mean_u=mean(D[tips,"u"]), mean_HWI=mean(D[tips,"HWI"]),
    mean_mass_g=10^mean(D[tips,"log_m"]))
}
R <- do.call(rbind, rows); write.csv(R, file.path(OUT,"S4_redo40.csv"), row.names=FALSE)
cat(sprintf("top40  r(delta_u, HWI) = %+.3f\n", cor(R$delta_u,R$mean_HWI)))
cat(sprintf("top20  r = %+.3f\n", cor(R$delta_u[1:20],R$mean_HWI[1:20])))
n<-R[R$delta_u<0,]; p<-R[R$delta_u>0,]
cat(sprintf("缩短 %d 支 HWI 均值 %.1f  |  伸长 %d 支 HWI 均值 %.1f\n",
  nrow(n),mean(n$mean_HWI),nrow(p),mean(p$mean_HWI)))
print(R[order(R$delta_u),c("delta_u","n_tip","level","label","purity","mean_HWI","mean_u","mean_mass_g")][1:12,],digits=3)
cat("\n--- 伸长的支系 ---\n")
print(R[R$delta_u>0,c("delta_u","n_tip","level","label","purity","mean_HWI","mean_u","mean_mass_g")],digits=3)
