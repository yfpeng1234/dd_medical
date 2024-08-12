library(dbnR)
library(here)
library(data.table)

#testing our optimized synthetic data
#first learn a DBN from it
#then compute the log-likelihood of test data

num_variable<-20
slices<-2

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

test1<-function(train_set){
  test_set<-read.csv(file.path(here(),'data','original_test_10.csv'))
  test_set<-test_set[,1:40]
  colnames(test_set)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))

  #learn DBN
  static_data<-copy(train_set[,1:(num_variable)])
  colnames(static_data)<-c(paste0("X", 1:num_variable, "_t_0"))
  net <- learn_dbn_struc(static_data, size=slices,f_dt=train_set)
  fitted_net<- fit_dbn_params(net = net,train_set)

  #use the learned DBN to compute the log-likelihood of test data
  score<-logLik(fitted_net,test_set)
  return(score)
}

test10<-function(dbn){
  test_set<-read.csv(file.path(here(),'data','original_test_10.csv'))

  #use the learned DBN to compute the log-likelihood of test data
  #for first 2 slices
  sub_test_set<-test_set[,1:(num_variable*slices)]
  colnames(sub_test_set)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
  score<-logLik(dbn,sub_test_set)

  #for other time slices
  node_to_consider<-colnames(sub_test_set)[(num_variable+1):(num_variable*slices)]
  for (i in 1:(10-2)){
    sub_test_set[,1:(num_variable*slices)]<-test_set[,(num_variable*i+1):(num_variable*i+num_variable*2)]
    score<-score+logLik(dbn,sub_test_set,nodes=node_to_consider)
  }
  return(score)
}

concat_test<-function(ipc){
  #concat
  original_data<-read.csv(file.path(here(),'data','original_train_10.csv'))
  original_data<-original_data[1:ipc,]
  concat_data<-original_data[,1:40]
  colnames(concat_data)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
  for (i in 1:8){
    add_data<-original_data[,((i*20+1):(i*20+40))]
    colnames(add_data)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
    concat_data<-rbind(concat_data,add_data)
  }
  score10<-test_10slices(concat_data)
  return (score10)
}

test_10slices<-function(synthetic_data){
  test_set<-read.csv(file.path(here(),'data','original_test_10.csv'))

  #learn DBN
  static_data<-copy(synthetic_data[,1:(num_variable)])
  colnames(static_data)<-c(paste0("X", 1:num_variable, "_t_0"))
  net <- learn_dbn_struc(static_data, size=slices,f_dt=synthetic_data)
  fitted_net<- fit_dbn_params(net = net,synthetic_data)

  #use the learned DBN to compute the log-likelihood of test data
  
  #for first 2 slices
  sub_test_set<-test_set[,1:(num_variable*slices)]
  colnames(sub_test_set)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
  score<-logLik(fitted_net,sub_test_set)

  #for other time slices
  node_to_consider<-colnames(sub_test_set)[(num_variable+1):(num_variable*slices)]
  for (i in 1:(10-2)){
    sub_test_set[,1:(num_variable*slices)]<-test_set[,(num_variable*i+1):(num_variable*i+num_variable*2)]
    score<-score+logLik(fitted_net,sub_test_set,nodes=node_to_consider)
  }
  return(score)
}

#use first 2 time slices
original_data<-read.csv(file.path(here(),'data','original_train_10.csv'))
original_data<-original_data[,1:40]
colnames(original_data)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
score10<-test_10slices(original_data[1:10,])
print(paste('use first 2 time slices:final log-likelihood(bigger->better) 10:',score10))
score20<-test_10slices(original_data[1:20,])
print(paste('use first 2 time slices:final log-likelihood(bigger->better) 20:',score20))
score50<-test_10slices(original_data[1:50,])
print(paste('use first 2 time slices:final log-likelihood(bigger->better) 50:',score50))
score100<-test_10slices(original_data[1:100,])
print(paste('use first 2 time slices:final log-likelihood(bigger->better) 100:',score100))

#use all slices and concat to 2 slices
score<-concat_test(1)
print(paste('use all slices and concat to 2 slices:final log-likelihood(bigger->better) 10:',score))
score<-concat_test(2)
print(paste('use all slices and concat to 2 slices:final log-likelihood(bigger->better) 20:',score))
score<-concat_test(5)
print(paste('use all slices and concat to 2 slices:final log-likelihood(bigger->better) 50:',score))
score<-concat_test(10)
print(paste('use all slices and concat to 2 slices:final log-likelihood(bigger->better) 100:',score))