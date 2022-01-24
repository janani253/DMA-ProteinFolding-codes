require(cluster)

args <- commandArgs(TRUE)
infile <- args[1]
outfile <- args[2]
nc <- as.integer(args[3])

d <- read.csv(infile, header=FALSE, sep=" ")
#l <- dim(d)[1]
#nc <- 500 #round(sqrt(l))   ##300


## clustering
pamd <- clara(d, nc, metric="manhattan")
#pamd <- pam(d, nc, metric="manhattan")


## Writing outputs
write(pamd$clustering, paste(outfile, "clusters", sep="."), ncol=1)
write(pamd$i.med, paste(outfile, "medoids", sep="."), ncol=1)
write.table(pamd$clusinfo, paste(outfile, "clusinfo", sep="."))

