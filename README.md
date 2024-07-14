# dd_medical
## Pipeline of synthetic data generation
1. run r/generate_data.R to get training set and test set
2. run python/process_data.py to split, add noise to, hide variables of training data
3. run r/dd.R to optimize synthetic data 
## note
1. **I guess cosine learning rate scheduler might be helpful. Intuitively, the optimization process should be periodically, first learning a better DBN structure, then learn a better parameter. We should use larger lr for the first period, smaller lr for the second period.**
2. **Currently lr=4e-5, sigma=0.1 is the best hyper parameter I found.**
