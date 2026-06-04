library(ggplot2)
library(viridisLite)
library(reshape2)

cat("Loading paper benchmark results...\n")
e <- readRDS("benchmark.results.rds")

cat("Loading Python package performance...\n")
py_res <- read.csv("python_benchmark_metrics.csv")
# Rename columns if needed to match e
# e has: dataset, method, AUPRC, AUROC, elapsed

for (i in 1:nrow(py_res)) {
  e <- rbind(e, c(py_res$dataset[i], py_res$method[i], py_res$AUPRC[i], py_res$AUROC[i], py_res$elapsed[i]))
}

# Ensure formats
e$AUPRC <- as.numeric(e$AUPRC)
e$AUROC <- as.numeric(e$AUROC)
e$elapsed <- as.numeric(e$elapsed)

cat("Calculating metrics and ranks (as per paper script)...\n")
datmax <- sort(apply(reshape2::dcast(e, method~dataset, value.var="AUPRC")[,-1], 2, na.rm=TRUE, FUN=max))
metmax <- sort(apply(reshape2::dcast(e, dataset~method, value.var="AUPRC")[,-1], 2, na.rm=TRUE, FUN=median))

# Customize ordering to put R/Python clustered then R/Python random at the top of the plot
met_levels <- names(metmax)
target_methods <- c("scDblFinder.Py.random", "scDblFinder.random", "scDblFinder.Py.clusters", "scDblFinder.clusters")
met_levels <- setdiff(met_levels, target_methods)
met_levels <- c(met_levels, target_methods)

e$dataset <- factor(e$dataset, levels=names(datmax))
e$method <- factor(e$method, levels=met_levels)
levels(e$method) <- gsub("bcds","scds::bcds",levels(e$method))
levels(e$method) <- gsub("cxds","scds::cxds",levels(e$method))
levels(e$method) <- gsub("hybrid","scds::hybrid",levels(e$method))

getranks <- function(x){
  y <- rank(x)
  y[is.na(x)] <- NA
  y
}

tr <- reshape2::dcast(e, method~dataset, value.var="AUPRC")
row.names(tr) <- tr[,1]; tr <- tr[,-1]
tr2 <- apply(tr, 2, FUN=getranks)

e$AUPRC.rank <- apply(e[,1:2], 1, FUN=function(x) tr2[as.character(x[2]),as.character(x[1])])
e$point.colour <- viridisLite::viridis(100)[round(100*e$AUPRC)]
e$border.colour <- ifelse(e$AUPRC.rank>=(length(metmax)-0.5), "black", NA)
e$rounded <- round(e$AUPRC,2)
e$text.colour <- ifelse(e$rounded>=0.5,"black","white")
e$text.colour[e$AUPRC.rank==1] <- NA
e$text <- gsub("1\\.00","1.0",gsub("0\\.",".",sprintf("%.2f",e$AUPRC)))

# The top methods to bold
scdbl.methods <- c("scDblFinder.clusters","scDblFinder.random", "scDblFinder.Py.clusters", "scDblFinder.Py.random", "directDblClassification","computeDoubletDensity")

p1 <- ggplot(e, aes(dataset, method)) + 
  geom_point(aes(size=AUPRC, colour=AUPRC.rank)) + 
  scale_size(range=c(4,10)) + 
  scale_colour_viridis_c(breaks=c(1, length(unique(e$method))), labels=c("worst","best"), guide=guide_colorbar(barheight=unit(1, "cm"))) + 
  geom_point(data=e[e$AUPRC.rank==length(unique(e$method)),], shape=21, colour="black", aes(size=AUPRC), stroke=1.1, show.legend=FALSE) + 
  geom_text(aes(label=text), size=3, colour=ifelse(e$AUPRC.rank>=5, "black", "white")) + 
  labs(colour="AUPRC rank") +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        axis.text.y=element_text(hjust=0.5, size=10.5, face="bold", 
                                 colour=ifelse(levels(e$method) %in% scdbl.methods,"black","grey30")),
        axis.title.y=element_blank(), panel.grid=element_blank())

ggsave("benchmark_AUPRC_fig.png", p1, width=10.5, height=6.8, dpi=300)
cat("Plot generated successfully!\n")
