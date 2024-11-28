import pandas as pd
import os

csv_folder = "/Users/pasqualecaggiano/Desktop/Master/Project/Bot-IoT/Dataset/CSVs"
csv_features = "/Users/pasqualecaggiano/Desktop/Master/Project/Bot-IoT/Dataset/UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv"

# Read the CSV with the headers
header_df = pd.read_csv(csv_features, header=None)  
headers = header_df.iloc[0].tolist() 
print(headers)
print(len(headers))

# Read and process each CSV file in the folder
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

dtype_dict = {5: str, 7: str}

dfs = []

for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    
    # Read the CSV without a header and apply the headers from the separate file
    df = pd.read_csv(file_path, header=None, dtype=dtype_dict)
    df.columns = headers  # Assign the headers to the dataframe
    
    # Add the dataframe to the list
    dfs.append(df)

# Concatenate all the dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

irrelevant_columns = {'pkSeqID',
                       'stime', 
                       'flgs', 
                       'proto', 
                       'saddr',
                       'sport', 
                       'daddr', 
                       'dport', 
                       'state', 
                       'ltime', 
                       'seq',
                       'smac',
                       'dmac',
                       'soui',
                       'doui',
                       'dco',
                       'sco'}

clean_df = combined_df.drop(columns=irrelevant_columns)

print(len(clean_df.columns))
print(clean_df.columns)