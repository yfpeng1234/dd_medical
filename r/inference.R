library(dbnR)
library(data.table)
library(bnlearn)
library(tictoc)

data <- read.csv("data.csv")
static_data<-copy(data[,c(1,2,3,4)])
old_names <- names(static_data)
new_names <- sub("_t_1$", "_t_0", old_names)
setnames(static_data,old=old_names,new = new_names)

net <- learn_dbn_struc(static_data, size=2,f_dt=data)
fitted_net<- fit_dbn_params(net = net,data)

result<- predict_dt(fit = fitted_net,dt=data,obj_nodes = c('d_t_1','b_t_0'),verbose = FALSE,look_ahead = TRUE)
