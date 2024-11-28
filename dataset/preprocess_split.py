import json
import os
import argparse


def preprocess_split(split_path, split_name, capture_name, split_type, train_val_split=0.8):

    split_file = os.path.join(split_path,split_name)

    with open(split_file, "r") as file:
        data = json.load(file)

    # Extract the list of graphs
    graph_lists = data[capture_name]

    # Flatten the list of lists into a single list
    all_graphs = [graph for sublist in graph_lists for graph in sublist]

    print(all_graphs)

    if split_type == "train_val":
        # Calculate the split index
        split_index = int(len(all_graphs) * train_val_split)

        # Split into train and val sets
        train_graphs = all_graphs[:split_index]
        val_graphs = all_graphs[split_index:]

        # Convert back to the original format (list of lists) if necessary
        train_data = {capture_name: train_graphs}
        val_data = {capture_name: val_graphs}

        train_path = os.path.join(split_path, "train.json")
        val_path = os.path.join(split_path, "val.json")

        # Save to train.json and val.json
        with open(train_path, "w") as train_file:
            json.dump(train_data, train_file, indent=4)

        with open(val_path, "w") as val_file:
            json.dump(val_data, val_file, indent=4)

        print(f"Files train.json and val.json have been preprocessed with an 80/20 split.")

    elif split_type == "test":
        # Create a new dictionary to hold the split data
        data = {capture_name: all_graphs}
        
        #split_file = os.path.join(split_path,"tmp.json")

        # Save the new dictionary to a new JSON file
        with open(split_file, "a") as output_file:
            json.dump(data, output_file, indent=4)

        print(f"File {split_name} has been preprocessed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_path', type=str, help="Path to the output folder")
    parser.add_argument('--split_name', type=str, help="Path to the JSON file to be preprocessed")
    parser.add_argument('--capture_name', type=str, help="Name of the capture to be preprocessed")
    parser.add_argument('--split_type', type=str, help="Type of split to be performed (train_val or test)")
    parser.add_argument('--train_val_split', type=float, default=0.8, help="Split ratio for train and validation sets")

    args = parser.parse_args()

    preprocess_split(args.split_path, args.split_name, args.capture_name, args.split_type, args.train_val_split)

