import pandas as pd
import numpy as np
import os

def rename_columns(data, num_features_per_time_slice=200, num_time_slices=2):
    
        new_columns = []
        for j in range(num_time_slices):
            new_columns.extend([f"X{k+1}_t_{num_time_slices-1-j}"  for k in range(num_features_per_time_slice)])
        
        # check that the length of new columns matches the number of DataFrame columns
        if len(new_columns) == data.shape[1]:
            data.columns = new_columns
        else:
            raise ValueError("The number of new column names does not match the number of columns in the DataFrame.")
        
        return data

def split(num_partitions=5,data=None):
    total_sample=data.shape[0]
    per_sample=total_sample//num_partitions
    L=[]
    for i in range(num_partitions):
        L.append(data.iloc[per_sample*i:per_sample*i+per_sample, :])
    return L

def add_noise_per_time_slice(data_list, num_features_per_time_slice=200, num_time_slices=2):
    
    L_withnoise = []
    for i in range(len(data_list)):
        df = data_list[i].copy()  
        for j in range(num_time_slices):
            
            #selecting the slice's data
            start_col = num_features_per_time_slice * j
            end_col = num_features_per_time_slice * (j + 1)
            one_slice = df.iloc[:, start_col:end_col]
            
            #generating the noise for every time slice 
            noise_cont = np.random.normal(loc=0, scale=0.1, size=one_slice.shape[0])
            
            for col in one_slice.columns[:]:  # ignoring the static variables
                df.iloc[:, start_col + one_slice.columns.get_loc(col)] += noise_cont
        
        L_withnoise.append(df)
    return L_withnoise

def hide_variables(data_list, num_features_to_hide, num_features_per_time_slice = 200, num_time_slices = 2):
    
    L_hidden = []

    for i in range(len(data_list)):
        df = data_list[i].copy()

        #### step 2: hide some time slices for the remaining variables ####
                
        # iterating over all the slices : 
        #to change with for j in range(1, num_time_slices-1) in case first and last slices shouldn't be included
        for j in range(num_time_slices): 
            df_slice = df.iloc[:,num_features_per_time_slice*j:num_features_per_time_slice*(j+1)]
            
            # choosing randomly the variables to hide 
            l = np.random.choice(num_features_per_time_slice, num_features_to_hide, replace=False) 
            # making sure that they aren't latent variables 
            valid_l = [elem for elem in l]
            df_slice.iloc[:, valid_l] = np.nan
            df.iloc[:,num_features_per_time_slice*j:num_features_per_time_slice*(j+1)] = df_slice    
        L_hidden.append(df)
    return L_hidden

def save(data_list):
    for i in range(len(data_list)):
        new = pd.DataFrame(data_list[i])
        if os.path.isdir("./../data/seperated_data") == False:
            os.mkdir("./../data/seperated_data")
        path = "./../data/seperated_data/partition_" + str(i) + ".csv"
        new.to_csv(path)

n=20
slices=2
partitions=5
num_hide=2

# reading data 
data_R_dbn = pd.read_csv("./../data/original_data.csv")
data_R_dbn = data_R_dbn.iloc[:,1:]

#rename the dataset to get the dbnR format
renamed_data= rename_columns(data_R_dbn,n,slices)

#split the data into 5 partitions to simulate multiple data sources
L=split(partitions,renamed_data)

#add noise to the data
L_withnoise=add_noise_per_time_slice(L,n,slices)

#hide some variables to simulate heterougeneous data type
L_hidden=hide_variables(L_withnoise,num_hide,n,slices)

#save the data
save(L_hidden)