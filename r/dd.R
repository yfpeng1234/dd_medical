#do we need moment to get out of sub-optimal no use
#do we need mini batch to choose part of synthetic data to optimize?    useful
#use real initialization is hard to optimize for dd_num=100
#seems that smaller sigma is better 8e-2 is the best
#what about only optimize hidden variables?
#for partial, 4e-4 lr is the best
#we can also try progressive dataset distillation, zero initialization for new data

library(dbnR)
library(here)
library(data.table)
library(ggplot2)
library(tictoc)

tic()

num_variable<-20
slices<-2
dd_sample<-20
epoch<-400
grad_add_num<-10
sigma<-1e-1
lr<-4e-5
seed<-44
partitions<-5
init_method<-'random'
optimizer<-'sgd'
#could be all, mini-batch,partial
#all -> optimize all the entry at the same time
#mini-batch -> optimize one source of the synthetic data at a time, with real initialization
#partial -> optimize only the hiden variables, with real initialization
sampling<-'all'

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
    if (sampling=='mini-batch'){
      num_per_batch<-dd_sample/partitions

      #initialize synthetic data from each data source
      idx<-1
      data_list<-list()
      root<-here()
      data_path<-file.path(root,'data','seperated_data')
      file_names<-list.files(path=data_path)
      for (name in file_names){
        df<-read.csv(file.path(data_path,name))
        df<-df[1:num_per_batch,]
        na_columns <- colnames(df)[apply(df, 2, function(col) all(is.na(col)))]
        na_columns_num<-length(na_columns)
        random_matrix <- matrix(rnorm(num_per_batch * na_columns_num), nrow = num_per_batch, ncol = na_columns_num)
        substitute_df<-as.data.frame(random_matrix)
        colnames(substitute_df)<-na_columns
        df[na_columns]<-substitute_df
        data_list[[idx]]<-df
        idx<-idx+1
      }
      #concat these data source
      random_df<-data_list[[1]]
      for (i in 2:partitions){
        random_df<-rbind(random_df,data_list[[i]])
      }
    }
    else if (sampling=='all'){
      random_df<-read.csv(file.path(here(),'data','seperated_data','partition_0.csv'))
      random_df<-random_df[1:dd_sample,]
      na_columns <- colnames(random_df)[apply(random_df, 2, function(col) all(is.na(col)))]
      na_columns_num<-length(na_columns)
      random_matrix <- matrix(rnorm(dd_sample * na_columns_num), nrow = dd_sample, ncol = na_columns_num)
      substitute_df<-as.data.frame(random_matrix)
      colnames(substitute_df)<-na_columns
      random_df[na_columns]<-substitute_df
    }
    else if (sampling=='partial'){
      random_df<-read.csv(file.path(here(),'data','seperated_data','partition_0.csv'))
      random_df<-random_df[1:dd_sample,]
    }
    else if (sampling=='stage'){
      random_df<-read.csv(file.path(here(),'data','seperated_data','partition_0.csv'))
      random_df<-random_df[1:dd_sample,]
    }
  }
  else if (init_method=='origin'){
    random_df<-read.csv(file.path(here(),'data','original_train.csv'))
    random_df<-random_df[1:dd_sample,]
  }
  return(random_df)
}

#evaluate the synthetic data by log-likelihood of partially observed training data
#synthetic_set: the synthetic data
#partial_set: the partially observed training data
eval_score<-function(synthetic_set,partial_set){
  #learn a DBN from synthetic data
  static_data<-copy(synthetic_set[,1:(num_variable)])
  colnames(static_data)<-c(paste0("X", 1:num_variable, "_t_0"))
  net <- learn_dbn_struc(static_data, size=slices,f_dt=synthetic_set)
  fitted_net<- fit_dbn_params(net = net,synthetic_set)
  
  #use the learned DBN to infer values of missing data
  test_set<-copy(partial_set)
  na_columns <- colnames(test_set)[apply(test_set, 2, function(col) all(is.na(col)))]
  result<- predict_dt(fit = fitted_net,dt=test_set,obj_nodes = na_columns,verbose = FALSE,look_ahead = TRUE)
  test_set[na_columns]<-result
  
  #compute the log-likelihood as the evaluation score, remember to use negative log-likelihood
  score<-logLik(fitted_net,test_set)
  return(-score)
}

#testing our optimized synthetic data
#first learn a DBN from it
#then compute the log-likelihood of test data
test<-function(synthetic_data){
  test_set<-read.csv(file.path(here(),'data','original_test.csv'))

  #learn DBN
  static_data<-copy(synthetic_data[,1:(num_variable)])
  colnames(static_data)<-c(paste0("X", 1:num_variable, "_t_0"))
  net <- learn_dbn_struc(static_data, size=slices,f_dt=synthetic_data)
  fitted_net<- fit_dbn_params(net = net,synthetic_data)

  #use the learned DBN to compute the log-likelihood of test data
  score<-logLik(fitted_net,test_set)
  return(score)
}

DD<-function(epochs=100,grad_add_num=10,sigma=0.5,lr=0.1){
  #save the test result each epoch
  LL<-c()

  #initialize our distilled data
  dd_data<-init_data()
  
  #get training data
  train_set<-get_dataset()
  num_train_set<-length(train_set)
  
  if (sampling=='mini-batch'){
    num_per_batch<-dd_sample/partitions
    for (i in 1:epochs){
      #test
      epoch_score<-test(dd_data)
      LL<-c(LL,epoch_score)

      choose_idx<-sample(1:num_train_set,1)
      choose_set<-train_set[[choose_idx]]
      
      #compute the gradient by zero order approximation
      grad<-matrix(0, nrow = num_per_batch, ncol = (num_variable*slices))
      score0<-eval_score(dd_data,choose_set)
      start_idx<-num_per_batch*(choose_idx-1)+1
      end_idx<-num_per_batch*choose_idx
      print(paste('epoch:',i,'  LL of test data(bigger->better):',epoch_score))
      for (j in 1:grad_add_num){
        random_perturb<-matrix(rnorm(num_per_batch * num_variable*slices), nrow = num_per_batch, ncol = num_variable*slices)
        perturbed_train_set<-dd_data
        perturbed_train_set[start_idx:end_idx,]<-perturbed_train_set[start_idx:end_idx,]+random_perturb*sigma
        score1<-eval_score(perturbed_train_set,choose_set)
        grad<-grad+random_perturb*(score1-score0)/sigma
      }
      grad<-grad/grad_add_num
      
      #update the synthetic data
      step_size<-lr
      dd_data[start_idx:end_idx,]<-dd_data[start_idx:end_idx,]-step_size*grad
    }
  }
  else if (sampling=='all' || init_method=='random'){
    for (i in 1:epochs){
      #test
      epoch_score<-test(dd_data)
      LL<-c(LL,epoch_score)

      choose_idx<-sample(1:num_train_set,1)
      choose_set<-train_set[[choose_idx]]
      
      #compute the gradient by zero order approximation
      grad<-matrix(0, nrow = dd_sample, ncol = (num_variable*slices))
      final_grad<-copy(grad)
      score0<-eval_score(dd_data,choose_set)
      print(paste('epoch:',i,'  LL of test data(bigger->better):',epoch_score))
      for (j in 1:grad_add_num){
        random_perturb<-matrix(rnorm(dd_sample * num_variable*slices), nrow = dd_sample, ncol = num_variable*slices)
        perturbed_train_set<-dd_data+random_perturb*sigma
        score1<-eval_score(perturbed_train_set,choose_set)
        grad<-grad+random_perturb*(score1-score0)/sigma
      }
      grad<-grad/grad_add_num
      final_grad<-0.9*final_grad+grad
      
      #update the synthetic data
      step_size<-lr
      if (optimizer=='sgd'){
        dd_data<-dd_data-step_size*grad
      }
      else if (optimizer=='momentum'){
        dd_data<-dd_data-step_size*final_grad
      }
    }
  }
  else if (sampling=='partial'){
    #only optimize the hidden variable
    na_columns <- colnames(dd_data)[apply(dd_data, 2, function(col) all(is.na(col)))]
    na_columns_num<-length(na_columns)
    random_matrix <- matrix(rnorm(dd_sample * na_columns_num), nrow = dd_sample, ncol = na_columns_num)
    substitute_df<-as.data.frame(random_matrix)
    colnames(substitute_df)<-na_columns
    dd_data[na_columns]<-substitute_df
    for (i in 1:epochs){
      #test
      epoch_score<-test(dd_data)
      LL<-c(LL,epoch_score)

      choose_idx<-sample(1:num_train_set,1)
      choose_set<-train_set[[choose_idx]]
      
      #compute the gradient by zero order approximation
      grad<-matrix(0, nrow = dd_sample, ncol = na_columns_num)
      score0<-eval_score(dd_data,choose_set)
      print(paste('epoch:',i,'  LL of test data(bigger->better):',epoch_score))
      for (j in 1:grad_add_num){
        random_perturb<-matrix(rnorm(dd_sample * na_columns_num), nrow = dd_sample, ncol = na_columns_num)
        perturbed_train_set<-dd_data
        perturbed_train_set[na_columns]<-perturbed_train_set[na_columns]+random_perturb*sigma
        score1<-eval_score(perturbed_train_set,choose_set)
        grad<-grad+random_perturb*(score1-score0)/sigma
      }
      grad<-grad/grad_add_num
      
      #update the synthetic data
      step_size<-lr
      dd_data[na_columns]<-dd_data[na_columns]-step_size*grad
    }
  }
  else if (sampling=='stage'){
    #first only optimize the hidden variable
    #then optimize all the data
    na_columns <- colnames(dd_data)[apply(dd_data, 2, function(col) all(is.na(col)))]
    na_columns_num<-length(na_columns)
    random_matrix <- matrix(rnorm(dd_sample * na_columns_num), nrow = dd_sample, ncol = na_columns_num)
    substitute_df<-as.data.frame(random_matrix)
    colnames(substitute_df)<-na_columns
    dd_data[na_columns]<-substitute_df
    for (i in 1:200){
      #test
      epoch_score<-test(dd_data)
      LL<-c(LL,epoch_score)

      choose_idx<-sample(1:num_train_set,1)
      choose_set<-train_set[[choose_idx]]
      
      #compute the gradient by zero order approximation
      grad<-matrix(0, nrow = dd_sample, ncol = na_columns_num)
      score0<-eval_score(dd_data,choose_set)
      print(paste('epoch:',i,'  LL of test data(bigger->better):',epoch_score))
      for (j in 1:grad_add_num){
        random_perturb<-matrix(rnorm(dd_sample * na_columns_num), nrow = dd_sample, ncol = na_columns_num)
        perturbed_train_set<-dd_data
        perturbed_train_set[na_columns]<-perturbed_train_set[na_columns]+random_perturb*sigma
        score1<-eval_score(perturbed_train_set,choose_set)
        grad<-grad+random_perturb*(score1-score0)/sigma
      }
      grad<-grad/grad_add_num
      
      #update the synthetic data
      step_size<-lr
      dd_data[na_columns]<-dd_data[na_columns]-step_size*grad

      #test
      epoch_score<-test(dd_data)
      LL<-c(LL,epoch_score)
    }

    #optimize all the data
    for (i in 201:epochs){
      choose_idx<-sample(1:num_train_set,1)
      choose_set<-train_set[[choose_idx]]
      
      #compute the gradient by zero order approximation
      grad<-matrix(0, nrow = dd_sample, ncol = (num_variable*slices))
      score0<-eval_score(dd_data,choose_set)
      print(paste('epoch:',i,'  choose data source:',choose_idx,'  loss:',score0))
      for (j in 1:grad_add_num){
        random_perturb<-matrix(rnorm(dd_sample * num_variable*slices), nrow = dd_sample, ncol = num_variable*slices)
        perturbed_train_set<-dd_data+random_perturb*sigma
        score1<-eval_score(perturbed_train_set,choose_set)
        grad<-grad+random_perturb*(score1-score0)/sigma
      }
      grad<-grad/grad_add_num

      #update the synthetic data
      step_size<-lr/10
      if (optimizer=='sgd'){
        dd_data<-dd_data-step_size*grad
      }
      else if (optimizer=='momentum'){
        dd_data<-dd_data-step_size*final_grad
      }
    }
  }
  

  #plot the test result
  idx<-1:epochs
  plot_data<-data.frame(epoch_idx=idx,LL_test=LL)
  ggplot(plot_data,aes(x=epoch_idx,y=LL_test))+geom_line()+xlab('Epoch')+ylab('Log-likelihood of test data')+ggtitle('Test result of distilled data')
  ggsave(file.path(here(),'result',paste('LL_on_test_set_epoch_',epochs,'_grad_add_num_',grad_add_num,'_sigma_',sigma,'_lr_',lr,'_dd_num_',dd_sample,'_init_',init_method,'_sampling_',sampling,'.png')),width=6,height=4)
  
  #save the synthetic data
  write.csv(dd_data,file=file.path(here(),'data','dd_data.csv'),row.names = FALSE)
}

DD(epochs = epoch,grad_add_num = grad_add_num,sigma=sigma,lr=lr)

toc()