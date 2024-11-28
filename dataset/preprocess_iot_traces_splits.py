import json
import os
import argparse

#list all the folders contained in a folder
def list_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def preprocess_split(split_path, split_name, captures):

    split_file = os.path.join(split_path,split_name)

    with open(split_file, "r") as file:
        data = json.load(file)

    graphs = {}
    #Flatten the list of lists for all the captures and obatain a single all_graph
    for capture_name in captures:
        # Check if key exists in the dictionary
        if capture_name not in data:
            print(f"Key '{capture_name}' not found in the dictionary.")
            continue
        all_graphs = data[capture_name]
        all_graphs = [item for sublist in all_graphs for item in sublist]
        graphs[capture_name] = all_graphs

    #print(graphs)

    # Save the new dictionary to a new JSON file
    with open(split_file, "w") as output_file:
        json.dump(graphs, output_file, indent=4)

    print(f"File {split_name} has been preprocessed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_path', type=str, help="Path to the output folder")
    parser.add_argument('--split_name', type=str, help="Path to the JSON file to be preprocessed")

    args = parser.parse_args()
    #id "Bot-IoT" is present in the split path
    capture = []
    if "Bot-IoT" in args.split_path:
        captures = list_folders("/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/Bot-IoT/60000/base")
    elif "IoT_traces" in args.split_path:
        captures = list_folders("/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT_traces/60000/base")
    
    print(captures)


    preprocess_split(args.split_path, args.split_name, captures)

