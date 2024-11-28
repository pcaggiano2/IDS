import os

def count_graphs(graph_folder):

    graph_types = [os.path.join(graph_folder,"tdg_graph"), os.path.join(graph_folder,"sim_graph")]

    stats_file = os.path.join(graph_folder, "graph_stats.txt")
    with open(stats_file, "w") as f:
        f.write("====================================\n")
        f.write(f"Folder: {graph_folder}\n")


    for type in graph_types:
        print(f"Processing type: {type.split('/')[-1]}")
        #count the number of files in the folder
        full_benign_count = 0
        full_malicious_count = 0
        mixed_count = 0

        for file in os.listdir(os.path.join(type, "full_benign")):
            if file.endswith(".pkl"):
                full_benign_count += 1

        for file in os.listdir(os.path.join(type, "full_malicious")):
            if file.endswith(".pkl"):
                full_malicious_count += 1

        for file in os.listdir(os.path.join(type, "mixed")):
            if file.endswith(".pkl"):
                mixed_count += 1

        #print the results
        print(f"Full benign: {full_benign_count}")
        print(f"Full malicious: {full_malicious_count}")
        print(f"Mixed: {mixed_count}")
        print(f'Total: {full_benign_count + full_malicious_count + mixed_count}')

        with open(stats_file, "a") as f:
            f.write(f"Type: {type.split('/')[-1]}\n")
            f.write(f"Full benign: {full_benign_count}\n")
            f.write(f"Full malicious: {full_malicious_count}\n")
            f.write(f"Mixed: {mixed_count}\n")
            f.write(f'Total: {full_benign_count + full_malicious_count + mixed_count}\n\n')

if __name__ == "__main__":

    graph_folders = ["/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT23/60000/base/single_capture_modified",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT23/90000/base/single_capture_modified",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT23/120000/base/single_capture_modified",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT_traces/60000/base",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT_traces/90000/base",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT_traces/120000/base",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoTID20/60000/base/single_capture_modified",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoTID20/90000/base/single_capture_modified",
                     "/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoTID20/120000/base/single_capture_modified"]

    for graph_folder in graph_folders:
        # check if in graph_folder is there a folder called "tdg_graph" and "sim_graph"
        if not os.path.isdir(os.path.join(graph_folder, "tdg_graph")) or not os.path.isdir(os.path.join(graph_folder, "sim_graph")):
            print(f"Folder {graph_folder} does not contain tdg_graph or sim_graph")
            #list all folder in graph_folder
            graph_folder = [os.path.join(graph_folder, d) for d in os.listdir(graph_folder) if os.path.isdir(os.path.join(graph_folder, d))]
            for graph_fol in graph_folder: 
                print(f"Processing folder: {graph_fol}")   
                count_graphs(graph_fol)
        else:
            print(f"Processing folder: {graph_folder}")
            count_graphs(graph_folder)

