suppressMessages(library(ape))
U<-"/mnt/user-data/uploads/FFY/FFY"; OUT<-"/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
tr0<-read.tree(file.path(U,"data/birdtree/hackett_tree1.nwk"))
D<-read.csv(file.path(U,"data/birdtree/pgls_data_full.csv")); rownames(D)<-D$tip
D<-D[!is.na(D$HWI),]
tr<-keep.tip(tr0,intersect(tr0$tip.label,D$tip)); D<-D[tr$tip.label,]
# 每个目取一代表种(u 最接近该目中位者)→ 真实子树,非示意
ords<-names(which(table(D$Order)>=10))
rep_tip<-sapply(ords,function(o){s<-D[D$Order==o,]; s$tip[which.min(abs(s$u-median(s$u)))]})
ot<-keep.tip(tr,as.character(rep_tip))
map<-setNames(names(rep_tip),as.character(rep_tip))
ot$tip.label<-map[ot$tip.label]
ot<-ladderize(ot)
pdf(NULL); pp<-plot.phylo(ot,plot=FALSE); env<-get("last_plot.phylo",envir=.PlotPhyloEnv); dev.off()
xx<-env$xx; yy<-env$yy
E<-data.frame(x0=xx[ot$edge[,1]],y0=yy[ot$edge[,1]],x1=xx[ot$edge[,2]],y1=yy[ot$edge[,2]])
write.csv(E,file.path(OUT,"tree_edges.csv"),row.names=FALSE)
TI<-data.frame(order=ot$tip.label,x=xx[1:Ntip(ot)],y=yy[1:Ntip(ot)])
st<-do.call(rbind,lapply(ords,function(o){s<-D[D$Order==o,]
  data.frame(order=o,n=nrow(s),u=median(s$u),HWI=median(s$HWI),mass=10^median(s$log_m))}))
TI<-merge(TI,st,by="order"); write.csv(TI,file.path(OUT,"tree_tips.csv"),row.names=FALSE)
# 把 40 个跳变事件挂到目上
S<-read.csv(file.path(OUT,"S4_redo40.csv"))
S$y<-TI$y[match(S$top_order,TI$order)]
write.csv(S[!is.na(S$y),],file.path(OUT,"S4_events_on_tree.csv"),row.names=FALSE)
cat("目数",nrow(TI)," 事件可定位",sum(!is.na(S$y)),"/",nrow(S),"\n")
cat("树高",max(xx),"\n")
