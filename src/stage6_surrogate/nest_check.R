suppressMessages(library(ape))
U<-"/mnt/user-data/uploads/FFY/FFY"
tr0<-read.tree(file.path(U,"data/birdtree/hackett_tree1.nwk"))
D<-read.csv(file.path(U,"data/birdtree/pgls_data_full.csv")); rownames(D)<-D$tip
D<-D[!is.na(D$HWI),]
tr<-keep.tip(tr0,intersect(tr0$tip.label,D$tip)); D<-D[tr$tip.label,]
y<-setNames(D$u,rownames(D)); a<-ace(y,tr,method="pic")$ace
st<-c(y[tr$tip.label],a); names(st)[1:Ntip(tr)]<-as.character(1:Ntip(tr))
ch<-st[as.character(tr$edge[,2])]-st[as.character(tr$edge[,1])]
pp<-prop.part(tr)
nd<-sapply(tr$edge[,2],function(v) if(v<=Ntip(tr)) 1L else length(pp[[v-Ntip(tr)]]))
sel<-which(nd>=20); ordr<-sel[order(-abs(ch[sel]))][1:40]
sets<-lapply(ordr,function(e){v<-tr$edge[e,2]; extract.clade(tr,v)$tip.label})
nest<-matrix(FALSE,40,40)
for(i in 1:40) for(j in 1:40) if(i!=j) nest[i,j]<- all(sets[[j]] %in% sets[[i]])
anc<-which(rowSums(nest)>0); desc<-which(colSums(nest)>0)
cat("40 个事件中:\n")
cat("  含有其它事件的(祖先支):",length(anc),"个 →",paste(sort(anc),collapse=","),"\n")
cat("  被其它事件包含的(嵌套支):",length(desc),"个 →",paste(sort(desc),collapse=","),"\n")
cat("  完全互不嵌套的:",40-length(union(anc,desc)),"个\n")
# 贪心取互不嵌套的最大集合(按|Δu|从大到小)
keep<-c()
for(i in 1:40){ ok<-TRUE
  for(k in keep) if(nest[i,k]||nest[k,i]) ok<-FALSE
  if(ok) keep<-c(keep,i) }
cat("\n按|Δu|优先贪心保留的互不嵌套事件:",length(keep),"个\n")
R<-read.csv("/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls/S4_redo40.csv")
R$independent<-FALSE; R$independent[keep]<-TRUE
write.csv(R,"/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls/S4_redo40_indep.csv",row.names=FALSE)
K<-R[keep,]
cat(sprintf("  其中缩短 %d 支(HWI 均值 %.1f) · 伸长 %d 支(%.1f) · r=%.3f\n",
  sum(K$delta_u<0),mean(K$mean_HWI[K$delta_u<0]),sum(K$delta_u>0),
  mean(K$mean_HWI[K$delta_u>0]),cor(K$delta_u,K$mean_HWI)))
