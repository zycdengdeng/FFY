# S3 复核:把「节点年龄」与「对比信息量(枝长)」解耦
# 关键:pic 的 variance = 该对比所用枝长之和。枝长短 → 噪声占比大 → 信号被稀释向 50%
suppressMessages(library(ape))
U<-"/mnt/user-data/uploads/FFY/FFY"
OUT<-"/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
D<-read.csv(file.path(U,"data/birdtree/pgls_data_full.csv")); rownames(D)<-D$tip
D<-D[!is.na(D$HWI),]
tr0<-read.tree(file.path(U,"data/birdtree/hackett_tree1.nwk"))
tr<-keep.tip(tr0,intersect(tr0$tip.label,D$tip)); D<-D[tr$tip.label,]

pu<-pic(setNames(D$u,rownames(D)),tr,var.contrasts=TRUE)
ph<-pic(setNames(D$HWI,rownames(D)),tr,var.contrasts=TRUE)
cu<-pu[,1]; ch<-ph[,1]; vv<-pu[,2]            # variance = 枝长之和(信息量)
ages<-branching.times(tr)[names(cu)]
opp<-(sign(cu)*sign(ch))<0

f<-function(z) if(length(z)<30) NA else mean(z)
p3<-function(z) if(length(z)<30) NA else binom.test(sum(z),length(z),.5)$p.value
qa<-quantile(ages,c(1/3,2/3)); qv<-quantile(vv,c(1/3,2/3))
A<-cut(ages,c(-Inf,qa,Inf),labels=c("浅","中","深"))
V<-cut(vv,  c(-Inf,qv,Inf),labels=c("短枝","中","长枝"))

cat("=== 单变量:年龄分层 (旧口径) ===\n")
for(l in levels(A)) cat(sprintf("  %s: %.1f%%  n=%d  p=%.1e\n",l,100*mean(opp[A==l]),sum(A==l),p3(opp[A==l])))
cat("\n=== 单变量:枝长(信息量)分层 ===\n")
for(l in levels(V)) cat(sprintf("  %s: %.1f%%  n=%d  p=%.1e\n",l,100*mean(opp[V==l]),sum(V==l),p3(opp[V==l])))
cat("\n=== 二维:枝长 × 年龄 (行=枝长,列=年龄) ===\n")
cat(sprintf("%8s %8s %8s %8s\n","","浅","中","深"))
for(lv in levels(V)){
  row<-sapply(levels(A),function(la) f(opp[V==lv & A==la]))
  nn <-sapply(levels(A),function(la) sum(V==lv & A==la))
  cat(sprintf("%8s %s\n",lv,paste(sprintf("%6s",ifelse(is.na(row),"  n<30",sprintf("%.1f%%",100*row))),collapse=" ")))
  cat(sprintf("%8s %s\n","(n)",paste(sprintf("%6d",nn),collapse=" ")))
}
cat("\n年龄与枝长的相关 r =", round(cor(ages,vv),3), "\n")
# 在固定枝长层内,年龄还有没有效应?
cat("\n=== 判定 ===\n")
for(lv in levels(V)){
  s<-opp[V==lv]; a2<-A[V==lv]
  if(sum(!is.na(a2))<90) next
  lo<-mean(s[a2=="浅"]); hi<-mean(s[a2=="深"])
  cat(sprintf("  %s 层内:浅 %.1f%% → 深 %.1f%%  (差 %+.1f pp)\n",lv,100*lo,100*hi,100*(hi-lo)))
}
