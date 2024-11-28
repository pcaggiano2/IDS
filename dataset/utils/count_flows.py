import pandas as pd
import os

def read_csv(csv_folder):
    # Read and process each CSV file in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    dfs = []

    for file in csv_files:
        file_path = os.path.join(csv_folder, file)
        
        # Read the CSV without a header and apply the headers from the separate file
        df = pd.read_csv(file_path)
        
        # Add the dataframe to the list
        dfs.append(df)

        # Concatenate all the dataframes into one
        df = pd.concat(dfs, ignore_index=True)

    return df


if __name__ == "__main__":
    csv_folders = ["/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets/Bot-IoT",
                   "/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets/IoT23",
                   "/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets/IoTID20",
                   "/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets/IoT_Traces"]

    for csv_folder in csv_folders:
        print(f"Processing folder: {csv_folder}")   

        df = read_csv(csv_folder)

        #count values in label column
        if csv_folder.split('/')[-1] == 'IoT_Traces':
            print(f"Benign\t {df['id'].count()}")
        else:
            print(df['detection_label'].value_counts())

    # gt_folder = "/Users/pasqualecaggiano/Desktop/Master/Project/OriginalDatasets/Bot-IoT/Ground_Truth"

    # # Initialize an empty Series to hold the summed counts
    # total_counts = pd.Series(dtype=int)

    # # Loop through files in the folder
    # for filename in os.listdir(gt_folder):
    #     file_path = os.path.join(gt_folder, filename)
        
    #     # Check if it is a file
    #     if os.path.isfile(file_path):
    #         print(f"Processing file: {filename}")
            
    #         # Read the CSV file
    #         df = pd.read_csv(file_path, sep=";")
            
    #         # Get the value counts for the 'attack' column
    #         attack_counts = df['attack'].value_counts()
    #         print(attack_counts)
    #         print('\n')
            
    #         # Add the value counts to the total (this will sum them)
    #         total_counts = total_counts.add(attack_counts, fill_value=0)

    # # Print the final summed counts
    # print("Total summed value counts for 'attack' across all files:")
    # print(total_counts)

