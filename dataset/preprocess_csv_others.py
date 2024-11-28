import pandas as pd
import argparse
import os

def preprocess_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Replace values in the "detection_label" column
    df['detection_label'] = df['detection_label'].replace({0: 'Benign', 1: 'Malicious'})

    # Save the modified DataFrame as a new CSV file
    df.to_csv(output_file, index=False)

    # Verify the new file was created
    print(f"New CSV file saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a CSV file.")
    parser.add_argument("--input_file", help="Path to the input CSV file.")
    parser.add_argument("--output_file", help="Path to the output CSV file.")
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} does not exist.")
        exit(1)

    # Check if the output file already exists
    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists.")
        exit(1)

    preprocess_csv(args.input_file, args.output_file)



