library(dbnR)
library(data.table)
library(tictoc)
tic()

data <- read.csv('D:/medical_data/DBN/orginal_data_renamed.csv')
static_data<-copy(data[,1:200])
old_names <- names(static_data)
new_names <- sub("_t_1$", "_t_0", old_names)
setnames(static_data,old=old_names,new = new_names)

net <- learn_dbn_struc(static_data, size=2,f_dt=data)

structure<-amat(net)
write.csv(structure,file = 'D:/medical_data/DBN/dmmhc_real.csv')

toc()