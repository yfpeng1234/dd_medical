############################################################################
### source : https://github.com/cbg-ethz/DBNclass + modifications 
#############################################################################

############################################################################
###This script is used for generating training and testing data
###to be more specific:
###1. Generate a random Gaussian DBN with 20 nodes and 2 slices (where 20 and 2 can be any other values)
###source nodes are Gaussian, N(0,0.5), while others are linear Gaussian N(f(parent_nodes),0.5),f(parent_nodes) is linear combination
###
###2. sampler 1000 observations respectively for training and testing set
###
###3. testing set is unchanged(do not add noise and hide variables)
###
###4. training set is fristly seperated to 5 partitions, each with 200 obervations
###
###5. each training partitions is added with Gaussian noise N(0,0.5), which can be controlled.
###then randomly hide 10% varibales , which can also be controlled
###
###step1-3 are done in this script, step4,5 are done in python/process_data.py
#############################################################################


library(Matrix)
library(data.table) 
library(pcalg)
library(BiDAG)
library(here)


nsamp = 1000
n = 20
slices = 10

DBNsimulation<-function(n, slices, nsamp=100) {
  root<- here()
  file_path<-file.path(root,'r','utils.R')
  source(file_path)
  DBN<-genDBN(1.3,n,slices,lB=0.4,uB=1.1)

  adj<- DBN$dbn
  wm<- DBN$wm

  #exporting weight matrix and adjacency matrix 
  fwrite(adj, file.path(root,'data','adjacent_matrix_10.csv'))
  fwrite(wm, file.path(root,'data','weight_matrix_10.csv'))

  #simulate data
  #generating original data for training and testing set
  simdata_train<-genDataDBN3(adj,wm,slices,ss=nsamp)
  simdata_train$mcmc<-simdata_train$mcmc[1:nsamp,]
  simdata_test<-genDataDBN3(adj,wm,slices,ss=nsamp)
  simdata_test$mcmc<-simdata_test$mcmc[1:nsamp,]
  
  #set names
  names<-c(paste0("X", 1:n, "_t_0"))
  for (i in 1:(slices-1)) {
    names<-c(paste0("X", 1:n, "_t_", i), names)
  }

  #exporting original data
  data_train<- as.data.frame(as.matrix(simdata_train$mcmc))
  colnames(data_train)<-names
  write.csv(data_train, file.path(root,'data','original_train_10.csv'),row.names = FALSE)
  data_test<- as.data.frame(as.matrix(simdata_test$mcmc))
  colnames(data_test)<-names
  write.csv(data_test, file.path(root,'data','original_test_10.csv'),row.names = FALSE)
  
}

## Optimal function to determine the order in the DAG 
orderdag_optim <- function(adj) {
  n <- ncol(adj)
  order <- integer(n)  # pre-allocate the order vector
  in_degree <- colSums(adj)  # calculate in-degrees of all nodes
  zero_in_deg_nodes <- which(in_degree == 0)
  cntr <- 1
  
  while (length(zero_in_deg_nodes) > 0) {
    current_node <- zero_in_deg_nodes[1]
    order[cntr] <- current_node
    cntr <- cntr + 1
    zero_in_deg_nodes <- zero_in_deg_nodes[-1]
    
    for (i in which(adj[current_node, ] != 0)) {
      in_degree[i] <- in_degree[i] - 1
      if (in_degree[i] == 0) {
        zero_in_deg_nodes <- c(zero_in_deg_nodes, i)
      }
    }
  }
  return(order)
}


## this one to get the DAG order for first slice and other slices separately 
orderdag3 <- function(adj, nsmall) {
  n <- ncol(adj)
  order_first_slice <- integer(0)  
  order_other_slices <- integer(0)  
  in_degree <- colSums(adj)
  zero_in_deg_nodes <- which(in_degree == 0)
  cntr <- 1
  
  while (length(zero_in_deg_nodes) > 0) {
    current_node <- zero_in_deg_nodes[1]
    zero_in_deg_nodes <- zero_in_deg_nodes[-1]
    
    if (current_node <= 2*nsmall) {
      # node is in the first slice
      order_first_slice <- c(order_first_slice, current_node)
    } else {
      # node is in other slices
      order_other_slices <- c(order_other_slices, current_node)
    }
    
    for (i in which(adj[current_node, ] != 0)) {
      in_degree[i] <- in_degree[i] - 1
      if (in_degree[i] == 0) {
        zero_in_deg_nodes <- c(zero_in_deg_nodes, i)
      }
    }
  }
  # Check if the graph is a DAG (i.e., all nodes are processed)
  if (length(order_first_slice) + length(order_other_slices) < n) {
    stop("not a DAG")
  } else {
    return(list(first_slice = order_first_slice, other_slices = order_other_slices))
  }
}

# generating the data from the DBN  
genDataDBN3<-function(adj,wm,  slices=2,ss=100) {
  # using Matrix sparse to avoid OOM crash 
  dbn<-Matrix(adj,sparse = TRUE)
  wm<-Matrix(wm,sparse = TRUE)
  n<-ncol(dbn)
  nsmall<-n/slices
  npar <- colSums(dbn)
  
  # for normal distributions 
  means<-rep(0,n)
  sigmas<-rep(0.5,n)
  
  #getting the order 
  orderx<-orderdag3(dbn, nsmall)
  
  #checking that the order is right and using it for now for the k loop 
  test_order <- orderdag_optim(dbn)
  rev_test_order <- rev(test_order)
  
  edges<-which(dbn!=0)
  nedges<-length(edges)
  #define order of a dag in first slice 
  ordery_first_slice <- orderx$first_slice
  
  datas <- Matrix(0, nrow = ss, ncol = n, sparse = TRUE)
  
  # generate data for the first time slice
  for(i in ordery_first_slice) {
    if(i %in% ordery_first_slice[(ordery_first_slice%in%c(1:nsmall))]){
      if(npar[i]==0) {
        datas[,i]<-rnorm(ss,mean=means[i])
      } 
      else {
        pari<-as.vector(which(dbn[,i]!=0))
        datas[,i]<-wm[pari,i] %*% t(datas[,pari])+rnorm(ss,mean=means[i],sd=sigmas[i])
      }
    }
    else{
      datas[,i]<-0
    }
    
  }
  
  #in case of slices more than slices = 2 all the other values are initialized to 0 
  datas[is.na(datas)]<-0
  
  dbn[,1:nsmall]<-0
  wm[,1:nsmall]<-0
  
  datal<-list()
  datal[[1]]<-datas
  
  # second loop to fill in the other time slices 
  for(k in 1:(slices-1)) {
    range1 <- 1:(nsmall * k)
    range2 <- (nsmall+nsmall*k+1):(slices*nsmall+1)
    ordery_k<-rev_test_order[!(rev_test_order%in%c(range1,range2))]
    if (length(ordery_k) == 0) {
      next  
    }
    datal[[k+1]]<-cbind(datal[[k]][,1:nsmall + (slices-1)*nsmall],matrix(0,nrow=ss,ncol=nsmall*(slices-1))) 
    for(i in ordery_k) {
      if(npar[i]==0) {
        datal[[k+1]][,i]<-rnorm(ss,mean=means[i])
      } 
      else {
        pari<-as.vector(which(dbn[,i]!=0))
        datal[[k+1]][,i]<-wm[pari,i] %*% t(datal[[k+1]][,pari])+rnorm(ss,mean=means[i],sd=sigmas[i])
      }
    }
  }
  
  if(slices>=2) {
    for(k in 1:(slices-1)) {
      datas <- datas+datal[[k+1]]
    }
  }
  res<-list()
  res$mcmc<-datas
  return(res)
}

DBNsimulation(n,slices,nsamp) # 100 time slices, 10 static variables, 200 variables and 1000 samples

