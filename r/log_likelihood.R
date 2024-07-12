library(bnlearn)
library(dbnR)
library(readr)

test_data = read.csv('/home/fadwa/Desktop/data_sets/time-series-data-generation/synthetic_datasets/5n_20ts_10N/test_data_5n_20ts_10N.csv',row.names = 1)
test_column_names <- colnames(test_data)

test_adj = read_csv('/home/fadwa/Downloads/test_MILPDBN_adj_5n_20ts_10N.csv')

test_m = as.matrix(test_adj)
#test_m[test_m!=0] <- 1

rownames(test_m) <- colnames(test_m) <- test_column_names

# Create an empty graph with the same nodes
test_bn <- empty.graph(nodes = test_column_names)

# Set the adjacency matrix to the bn object
amat(test_bn) <- test_m

test_net = test_bn
class(test_net) <- c("dbn", class(test_net))

logLik(test_net, test_data)
