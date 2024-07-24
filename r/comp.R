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

#full original training set
original_data<-read.csv(file.path(here(),'data','original_train.csv'))
score1<-test(original_data)
print(paste('original full training set:final log-likelihood(bigger->better):',score1))

#subset of original training set (100)
original_data<-read.csv(file.path(here(),'data','original_train.csv'))
original_data<-original_data[1:100,]
score2<-test(original_data)
print(paste('original subset(100) training set:final log-likelihood(bigger->better):',score2))

#subset of original training set (10)
original_data<-read.csv(file.path(here(),'data','original_train.csv'))
original_data<-original_data[1:10,]
score3<-test(original_data)
print(paste('original subset(10) training set:final log-likelihood(bigger->better):',score3))

#synthetic data
synthetic_data<-read.csv(file.path(here(),'data','dd_data.csv'))
score4<-test(synthetic_data)
print(paste('synthetic data:final log-likelihood(bigger->better):',score4))

#mads
mad_optimized_data<-read.csv(file.path(here(),'data','mads_optimized_dd_data.csv'))
score5<-test(mad_optimized_data)
print(paste('mads optimized data:final log-likelihood(bigger->better):',score5))

#random noise
synthetic_data<-matrix(rnorm(10*40), nrow = 10, ncol = 40)
synthetic_data<-as.data.frame(synthetic_data)
colnames(synthetic_data)<-c(paste0("X", 1:num_variable, "_t_1"), paste0("X", 1:num_variable, "_t_0"))
score6<-test(synthetic_data)
print(paste('random noise:final log-likelihood(bigger->better):',score6))

#original data with noise , no hidden variables (100)
synthetic_data<-read.csv(file.path(here(),'data','original_train.csv'))
synthetic_data<-synthetic_data[1:100,]
synthetic_data<-synthetic_data+matrix(rnorm(100*40), nrow = 100, ncol = 40)*0.1
score7<-test(synthetic_data)
print(paste('original data with noise:final log-likelihood(bigger->better):',score7))

#subset of original data with noise , no hidden variables (20)
synthetic_data<-read.csv(file.path(here(),'data','original_train.csv'))
synthetic_data<-synthetic_data[1:20,]
score8<-test(synthetic_data)
print(paste('original subset(20) training set:final log-likelihood(bigger->better):',score8))

#subset of original data with noise , no hidden variables (50)
synthetic_data<-read.csv(file.path(here(),'data','original_train.csv'))
synthetic_data<-synthetic_data[1:50,]
score9<-test(synthetic_data)
print(paste('original subset(50) training set:final log-likelihood(bigger->better):',score9))