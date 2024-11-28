import csv
import debugpy
import argparse
import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    old_id_to_new_id_dict = dict()

    # Step 1 - Create a file named elliptic_txs_orig2contiguos.csv and modify elliptic_txs_features.csv.
    with open('data/elliptic_bitcoin_dataset/elliptic_txs_features.csv', 'r') as csv_file:
        with open('data/elliptic_bitcoin_dataset/elliptic_txs_features_modified.csv', 'w', encoding='UTF8') as input_writer:
            with open('data/elliptic_bitcoin_dataset/elliptic_txs_orig2contiguos.csv', 'w', encoding='UTF8') as csv_writer:

                # write head of output file
                csv_writer = csv.writer(csv_writer, delimiter=',')
                header = ["originalId", "contiguosId"]
                csv_writer.writerow(header)

                # input writer
                input_writer = csv.writer(input_writer, delimiter=',')

                csv_reader = csv.reader(csv_file, delimiter=',')

                for row_number, row in enumerate(csv_reader):
                    print(row_number)
                    # take node_id
                    node_id = row[0]
                    old_id_to_new_id_dict[node_id] = row_number
                    # write association between old_id and new id
                    out = [node_id, row_number]
                    csv_writer.writerow(out)

                    # update input file
                    row[0] = float(row_number)
                    row[1] = float(row[1])
                    input_writer.writerow(row)

    # Step 2 - Modify elliptic_txs_classes.csv
    with open('data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        with open('data/elliptic_bitcoin_dataset/elliptic_txs_classes_modified.csv', 'w', encoding='UTF8') as input_writer:
            # input writer
            input_writer = csv.writer(input_writer, delimiter=',')

            # write head
            input_writer.writerow(["txId", "timestep"])
            for row_number, row in enumerate(csv_reader):
                print(row)
                if row_number == 0:
                    print(row)
                else:
                    # substitute node_id with row_number
                    row[0] = float(row_number-1)

                    # change class
                    if row[1] == "unknown":
                        row[1] = float(-1.0)
                    elif int(row[1]) == 1:
                        row[1] = float(1.0)
                    elif int(row[1]) == 2:
                        row[1] = float(0.0)
                    input_writer.writerow(row)

    # Step 3 -  Create a file named elliptic_txs_nodetime.csv
    node_id_to_timestamp_dict = dict()
    with open('data/elliptic_bitcoin_dataset/elliptic_txs_features.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('data/elliptic_bitcoin_dataset/elliptic_txs_nodetime.csv', 'w', encoding='UTF8') as csv_writer:
            # input writer
            csv_writer = csv.writer(csv_writer, delimiter=',')

            # write header
            csv_writer.writerow(["txId", "timestep"])

            for row_number, row in enumerate(csv_reader):
                print(row_number)
                time_stamp = float(row[1])-1
                new_row = [row_number, time_stamp]
                # write row
                csv_writer.writerow(new_row)
                node_id_to_timestamp_dict[row[0]] = time_stamp

    # Step 4 - Modify elliptic_txs_edgelist.csv and rename it to elliptic_txs_edgelist_timed.csv
    with open('data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('data/elliptic_bitcoin_dataset/elliptic_txs_edgelist_timed.csv', 'w', encoding='UTF8') as csv_writer:
            # input writer
            csv_writer = csv.writer(csv_writer, delimiter=',')

            # write header
            csv_writer.writerow(["txId1", "txId2", "timestep"])

            for row_number, row in enumerate(csv_reader):
                print(row_number)
                if row_number == 0:
                    pass
                else:
                    # change node id with new id
                    assert node_id_to_timestamp_dict[row[0]] == node_id_to_timestamp_dict[row[1]
                                                                                          ], f"Timestamp of nodes {old_id_to_new_id_dict[row[0]]} and {old_id_to_new_id_dict[row[1]]} must be equal"
                    csv_writer.writerow(
                        [old_id_to_new_id_dict[row[0]], old_id_to_new_id_dict[row[1]], float(node_id_to_timestamp_dict[row[0]])])
