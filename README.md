# dd_medical
## Pipeline of synthetic data generation
1. run r/generate_data.R to get training set and test set
2. run python/process_data.py to split, add noise to, hide variables of training data
3. run r/dd.R to optimize synthetic data 

## Result
|   IPC   | original data   | synthetic data   | 
|:------:|:------:|:------:|
| 10  | -58085.4476498256 | -43299.6814439496 |
| 20  | -42319.0829008116 | -39785.7854257837 |
| 50  | -38384.9103375109 | -37756.202634252 |
| 100  | -36560.854693464 | -36338.4319040708 |
| all  | -35210.0823151876 | - |