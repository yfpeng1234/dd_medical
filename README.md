# Medical data
## Pipeline of synthetic data generation
1. run r/generate_data.R to get training set and test set
2. run python/process_data.py to split, add noise to, hide variables of training data
3. run r/dd.R to optimize synthetic data
4. run r/dd_multi_slices.R to run dd for scaling up time-slices setting 

## Main Result
|   IPC   | original data   | synthetic data   | 
|:------:|:------:|:------:|
| 10  | -58085.4476498256 | -43299.6814439496 |
| 20  | -42319.0829008116 | -39785.7854257837 |
| 50  | -38384.9103375109 | -37756.202634252 |
| 100  | -36560.854693464 | -36338.4319040708 |
| all  | -35210.0823151876 | - |

## Scaling time-slices
|   IPC   | original data   | synthetic data   | 
|:------:|:------:|:------:|
| 10  | -232776.9566 | -178320.4405 |
| 20  | -180134.7599 | -156528.6462 |
| 50  | -162529.1745 | -154271.8604 |
| 100  | -155691.3231 | -154570.5172 |
| all  | -151192.2361 | - |

## Hyperparameter
Please refer to the comments in r/dd.R and r/dd_multi_slices.R

# Boundary Generalizable PINN
## Running command
```bash
cd deepxde
python dfo.py
```

## Result
|   IPC   | budget |original data   | synthetic data   | 
|:------:|:------:|:------:|:------:|
| 5  | 300 |0.36997312 | 0.17892931 |
| 10  | 300| 0.34590203 | 0.15414341 |
| 20  | 400|0.098028466 | 0.07546207 |
| 40  | 100|0.040654637 | 0.040302712 |
