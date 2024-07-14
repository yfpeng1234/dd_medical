# dd_medical
## Pipeline of synthetic data generation
1. run r/generate_data.R to get training set and test set
2. run python/process_data.py to split, add noise to, hide variables of training data
3. run r/dd.R to optimize synthetic data 
