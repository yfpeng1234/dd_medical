library(dbnR)
library(here)
library(data.table)

num_variable<-20
slices<-2
dd_sample<-100
epoch<-100
grad_add_num<-10
sigma<-0.1
lr<-0.1

get_dataset<-function(){
  data_list<-list()
  idx<-1
  
  root<-here()
  data_path<-file.path(root,'data','seperated_data')
  file_names<-list.files(path=data_path)
  
  for (name in file_names){
    df<-read.csv(file.path(data_path,name))
    data_list[idx]<-df
    idx<-idx+1
  }
  
  return(data_list)
}

init_data<-function(){
  set.seed(123)
  random_matrix <- matrix(rnorm(dd_sample * num_variable*slices), nrow = dd_sample, ncol = num_variable*slices)
  random_df <- as.data.frame(random_matrix)
  colnames(random_df) <- c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
  return(random_df)
}

eval_score<-function(train_set,partial_set){
  #learn a DBN from synthetic data
  static_data<-copy(train_set[,1:(num_variable)])
  old_name<-c(paste0("X", 1:num_variable, "_t_1"))
  new_name<-c(paste0("X", 1:num_variable, "_t_0"))
  setnames(static_data,old=old_name,new = new_name)
  net <- learn_dbn_struc(static_data, size=slices,f_dt=train_set)
  fitted_net<- fit_dbn_params(net = net,train_set)
  
  #use the learned DBN to infer values of missing data, don't know how to use EM here
  test_set<-copy(partial_set)
  na_columns <- colnames(test_set)[apply(test_set, 2, function(col) all(is.na(col)))]
  result<- predict_dt(fit = fitted_net,dt=test_set,obj_nodes = na_columns,verbose = FALSE,look_ahead = TRUE)
  test_set[na_columns]<-result
  
  #compute the log-likelihood as the evaluation score
  score<-logLik(fitted_net,test_set)
  return(score)
}

dmmhc<-function(train_set){
  #learn a DBN from synthetic data
  static_data<-copy(train_set[,1:(num_variable)])
  old_name<-c(paste0("X", 1:num_variable, "_t_1"))
  new_name<-c(paste0("X", 1:num_variable, "_t_0"))
  setnames(static_data,old=old_name,new = new_name)
  net <- learn_dbn_struc(static_data, size=slices,f_dt=train_set)
  structure<-amat(net)
  return(structure)
}

DD<-function(epochs=100,grad_add_num=10,sigma=0.5,lr=0.1){
  #initialize our distilled data
  dd_data<-init_data()
  
  #get training data
  train_set<-get_dataset()
  num_train_set<-length(train_set)
  
  for (i in 1:epochs){
    choose_idx<-sample(1:num_train_set,1)
    choose_set<-train_set[choose_idx]
    print(paste('epoch:',i,'  choose data source:',choose_idx))
    
    #compute the gradient by zero order approximation
    grad<-matrix(0, nrow = dd_sample, ncol = (num_variable*slices))
    score0<-eval_score(dd_data,choose_set)
    for (j in 1:grad_add_num){
      random_perturb<-matrix(rnorm(dd_sample * num_variable*slices), nrow = dd_sample, ncol = num_variable*slices)
      perturbed_train_set<-dd_data+random_perturb*sigma()
      score1<-eval_score(perturbed_train_set,choose_set)
      grad<-grad+random_perturb*(score1-score0)/sigma()
    }
    grad<-grad/grad_add_num
    
    #update the synthetic data
    step_size<-lr/sqrt(i)
    dd_data<-dd_data-step_size*grad
  }
  
  #learn the structure from synthetic data and export it
  structure<-dmmhc(dd_data)
  write.csv(structure,file = file.path(here(),'data','adj.csv'))
  
  #save the synthetic data
  write.csv(dd_data,file=file.path(here(),'data','dd_data.csv'))
}

DD(epochs = epoch,grad_add_num = grad_add_num,sigma=sigma,lr=lr)