library(dbnR)
library(here)
library(data.table)
library(ggplot2)
library(tictoc)
library(dfoptim)

tic()

num_variable<-20
slices<-2
dd_sample<-100
epoch<-160
grad_add_num<-10
sigma<-0.1
lr<-4e-5
seed<-44
init_method<-'real'

#set seed for reproduction
set.seed(seed)

get_dataset<-function(){
  data_list<-list()
  idx<-1
  
  root<-here()
  data_path<-file.path(root,'data','seperated_data')
  file_names<-list.files(path=data_path)
  
  for (name in file_names){
    df<-read.csv(file.path(data_path,name))
    data_list[[idx]]<-df
    idx<-idx+1
  }
  
  return(data_list)
}

init_data<-function(){
  if (init_method=='random'){
    random_matrix <- matrix(rnorm(dd_sample * num_variable*slices), nrow = dd_sample, ncol = num_variable*slices)
    random_df <- as.data.frame(random_matrix)
    colnames(random_df) <- c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
  }
  else if (init_method=='real'){
    random_df<-read.csv(file.path(here(),'data','seperated_data','partition_0.csv'))
    random_df<-random_df[1:dd_sample,]
    na_columns <- colnames(random_df)[apply(random_df, 2, function(col) all(is.na(col)))]
    na_columns_num<-length(na_columns)
    random_matrix <- matrix(rnorm(dd_sample * na_columns_num), nrow = dd_sample, ncol = na_columns_num)
    substitute_df<-as.data.frame(random_matrix)
    colnames(substitute_df)<-na_columns
    random_df[na_columns]<-substitute_df
  }
  return(random_df)
}

#initial point of optimization variables
init_value<-init_data()
#convert format from data frame to vector
init_value<-as.vector(as.matrix(init_value))

#get training set
train_set<-get_dataset()
num_train_set<-length(train_set)

#objective function
obj<-function(x){
  #convert format
  dd_data<-as.data.frame(matrix(x, nrow = dd_sample, ncol = num_variable*slices))
  colnames(dd_data) <- c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))

  #learn a DBN from synthetic data (inner loop)
  static_data<-copy(dd_data[,1:(num_variable)])
  colnames(static_data)<-c(paste0("X", 1:num_variable, "_t_0"))
  net <- learn_dbn_struc(static_data, size=slices,f_dt=dd_data)
  fitted_net<- fit_dbn_params(net = net,dd_data)

  #compute the log-likelihood of training data
  Neg_LL_on_train<-0
  for (i in 1:num_train_set){
    #use the learned DBN to infer values of missing data
    test_set<-copy(train_set[[i]])
    na_columns <- colnames(test_set)[apply(test_set, 2, function(col) all(is.na(col)))]
    result<- predict_dt(fit = fitted_net,dt=test_set,obj_nodes = na_columns,verbose = FALSE,look_ahead = TRUE)
    test_set[na_columns]<-result
  
    #compute the log-likelihood as the evaluation score, remember to use negative log-likelihood
    score<-logLik(fitted_net,test_set)
    Neg_LL_on_train<-Neg_LL_on_train-score
  }
  return (Neg_LL_on_train)
}

#start optimization
#best deltaInit=1e-4 epoch=2000 LL=-49797
result<-mads(par=init_value,fn=obj,control=list(maxfeval=4000,tol=1e-6,deltaInit=1e-5,expand=4,lineSearch=20))
optimized_dd_data<-result$par
#save the result
optimized_dd_data<-as.data.frame(matrix(optimized_dd_data, nrow = dd_sample, ncol = num_variable*slices))
colnames(optimized_dd_data) <- c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
write.csv(optimized_dd_data,file.path(here(),'data','mads_optimized_dd_data.csv'),row.names = FALSE)


toc()