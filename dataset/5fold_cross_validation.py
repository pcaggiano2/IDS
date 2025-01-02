import json
from sklearn.model_selection import KFold
import os
import argparse

def create_folds(file_path, output_dir, n_splits=5):
    # Carica il file JSON
    with open(file_path, 'r') as file:
        data = json.load(file)

    for key, samples in data.items():
        kf = KFold(n_splits=n_splits, shuffle=False)
        samples = list(samples)
        
        # Itera sui fold
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(samples), 1):
            train_samples = [samples[i] for i in train_indices]
            val_samples = [samples[i] for i in val_indices]

            # Prepara i file di output
            fold_train_file = os.path.join(output_dir, f"train{fold_idx}.json")
            fold_val_file = os.path.join(output_dir, f"val{fold_idx}.json")

            # Salva i file JSON
            with open(fold_train_file, 'w') as train_file:
                json.dump({key: train_samples}, train_file, indent=4)
            with open(fold_val_file, 'w') as val_file:
                json.dump({key: val_samples}, val_file, indent=4)

            print(f"Fold {fold_idx} salvato: {fold_train_file}, {fold_val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create folds from a JSON file.")
    parser.add_argument("--input_file", default="/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/train_val.json", help="Path to the input JSON file.")
    parser.add_argument("--output_directory",default="/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/5folds", help="Path to the output directory.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds.")

    args = parser.parse_args()
    print(args.input_file)
    print(args.output_directory)

    create_folds(args.input_file, args.output_directory, args.n_splits)