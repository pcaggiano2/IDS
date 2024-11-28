from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as u
import node_cls_tasker as nct
import node_anomaly_tasker as nat
from collections import OrderedDict
from torch.utils.data import SubsetRandomSampler
import os
import json


def collate_func(data):
    return data


class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''

    def __init__(self, args, tasker):

        if tasker.is_static:  # For static datsets
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            # only the training one requires special handling on start, the others are fine with the split IDX.

            random_perm = False
            indexes = tasker.data.nodes_with_label

            if random_perm:
                perm_idx = torch.randperm(indexes.size(0))
                perm_idx = indexes[perm_idx]
            else:
                print('tasker.data.nodes', indexes.size())
                perm_idx, _ = indexes.sort()
            # print ('perm_idx',perm_idx[:10])

            self.train_idx = perm_idx[:int(
                args.train_proportion*perm_idx.size(0))]
            self.dev_idx = perm_idx[int(args.train_proportion*perm_idx.size(0)): int(
                (args.train_proportion+args.dev_proportion)*perm_idx.size(0))]
            self.test_idx = perm_idx[int(
                (args.train_proportion+args.dev_proportion)*perm_idx.size(0)):]
            # print ('train,dev,test',self.train_idx.size(), self.dev_idx.size(), self.test_idx.size())

            train = static_data_split(tasker, self.train_idx, test=False)
            train = DataLoader(train, shuffle=True, **args.data_loading_params)

            dev = static_data_split(tasker, self.dev_idx, test=True)
            dev = DataLoader(dev, shuffle=False, **args.data_loading_params)

            test = static_data_split(tasker, self.test_idx, test=True)
            test = DataLoader(test, shuffle=False, **args.data_loading_params)

            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test

        else:  # For datsets with time
            if isinstance(tasker, nct.Node_Cls_Tasker):
                assert args.train_proportion + args.dev_proportion < 1, \
                    'there\'s no space for test samples'
                # only the training one requires special handling on start, the others are fine with the split IDX.
                start = tasker.data.min_time + args.num_hist_steps  # -1 + args.adj_mat_time_window
                end = args.train_proportion

                end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
                train = data_split(tasker, start, end, test=False)
                train = DataLoader(train, **args.data_loading_params)

                start = end
                end = args.dev_proportion + args.train_proportion
                end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
                if args.task == 'link_pred':
                    dev = data_split(tasker, start, end,
                                     test=True, all_edges=True)
                else:
                    dev = data_split(tasker, start, end, test=True)

                dev = DataLoader(
                    dev, num_workers=args.data_loading_params['num_workers'])

                start = end

                # the +1 is because I assume that max_time exists in the dataset
                end = int(tasker.max_time) + 1
                if args.task == 'link_pred':
                    test = data_split(tasker, start, end,
                                      test=True, all_edges=True)
                else:
                    test = data_split(tasker, start, end, test=True)

                test = DataLoader(
                    test, num_workers=args.data_loading_params['num_workers'])

                print('Dataset splits sizes:  train', len(
                    train), 'dev', len(dev), 'test', len(test))

            elif isinstance(tasker, nat.Anomaly_Detection_Tasker):
                # train, dev, test = self.anomaly_split(tasker, args)
                train = AnomalyDataset(
                    tasker=tasker,
                    path=tasker.data.dataset_path,
                    split='train')
                print(f"Train len {len(train)}")
                dev = AnomalyDataset(tasker=tasker,
                                     path=tasker.data.dataset_path,
                                     split='val')
                print(f"Val len {len(dev)}")

                if not tasker.data.sequence:
                    test_benign = AnomalyDataset(tasker=tasker,
                                                 path=tasker.data.dataset_path,
                                                 split='test_benign')
                    print(f"test_benign len {len(test_benign)}")
                    test_mixed = AnomalyDataset(tasker=tasker,
                                                path=tasker.data.dataset_path,
                                                split='test_mixed')
                    print(f"test_mixed len {len(test_mixed)}")
                    test_malicious = AnomalyDataset(tasker=tasker,
                                                    path=tasker.data.dataset_path,
                                                    split='test_malicious')
                    print(f"test_malicious len {len(test_malicious)}")
                else:
                    test_iot23 = AnomalyDataset(tasker=tasker,
                                                path=tasker.data.dataset_path,
                                                split="test_iot23")
                    print(f"test_iot23 len {len(test_iot23)}")

                test_traces = AnomalyDataset(tasker=tasker,
                                             path=tasker.data.iot_traces_path,
                                             split='IoT_traces')
                print(f"test_traces len {len(test_traces)}")

                if not tasker.data.sequence:

                    test_iotid20_benign = AnomalyDataset(tasker=tasker,
                                                         path=tasker.data.iot_id20_path,
                                                         split='IoTID20_benign')
                    print(f"test_iotid20 len {len(test_iotid20_benign)}")
                    test_iotid20_mixed = AnomalyDataset(tasker=tasker,
                                                        path=tasker.data.iot_id20_path,
                                                        split='IoTID20_mixed')
                    print(f"test_iotid20 len {len(test_iotid20_mixed)}")
                else:
                    test_iotid20 = AnomalyDataset(tasker=tasker,
                                                  path=tasker.data.iot_id20_path,
                                                  split="IoTID20")
                    print(f"test_iotid20 len {len(test_iotid20)}")

                train = DataLoader(
                    train, shuffle=True, num_workers=args.data_loading_params['num_workers'], batch_size=args.data_loading_params['batch_size'], collate_fn=collate_func)
                dev = DataLoader(
                    dev, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                if not tasker.data.sequence:
                    test_benign = DataLoader(
                        test_benign, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    test_mixed = DataLoader(
                        test_mixed, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    test_malicious = DataLoader(
                        test_malicious, shuffle=False,  num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                else:
                    test_iot23 = DataLoader(
                        test_iot23, shuffle=False,  num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)

                test_traces = DataLoader(
                    test_traces, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)

                if not tasker.data.sequence:
                    test_iotid20_benign = DataLoader(
                        test_iotid20_benign, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    test_iotid20_mixed = DataLoader(
                        test_iotid20_mixed, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                else:
                    test_iotid20 = DataLoader(
                        test_iotid20, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
            elif isinstance(tasker, nat.Anomaly_Detection_Tasker_IoT_traces):
                # train, dev, test = self.anomaly_split(tasker, args)
                train = AnomalyDataset(
                    tasker=tasker,
                    path=tasker.data.dataset_path,
                    split='train')
                print(f"Train len {len(train)}")
                dev = AnomalyDataset(tasker=tasker,
                                     path=tasker.data.dataset_path,
                                     split='val')
                print(f"Val len {len(dev)}")

                if not tasker.data.sequence:
                    test_benign = AnomalyDataset(tasker=tasker,
                                                 path=tasker.data.dataset_path,
                                                 split='test_benign_iot_traces')
                    print(f"test_benign len {len(test_benign)}")
                    
                    
                    #test_iot23_benign = AnomalyDataset(tasker=tasker,
                    #                                     path=tasker.data.iot23_path,
                    #                                     split='test_benign')
                    #print(f"test_iot23 benign len {len(test_iot23_benign)}")
                    test_iot23_mixed = AnomalyDataset(tasker=tasker,
                                                        path=tasker.data.iot23_path,
                                                        split='test_mixed')
                    print(f"test_iot23 mix len {len(test_iot23_mixed)}")
                    
                    test_iot23_malicious = AnomalyDataset(tasker=tasker,
                                                        path=tasker.data.iot23_path,
                                                        split='test_malicious')
                    print(f"test_iot23 mal len {len(test_iot23_malicious)}")
                    
                    
                    test_iotid20_benign = AnomalyDataset(tasker=tasker,
                                                         path=tasker.data.iot_id20_path,
                                                         split='IoTID20_benign')
                    print(f"test_iotid20 len {len(test_iotid20_benign)}")
                    test_iotid20_mixed = AnomalyDataset(tasker=tasker,
                                                        path=tasker.data.iot_id20_path,
                                                        split='IoTID20_mixed')
                    print(f"test_iotid20 len {len(test_iotid20_mixed)}")
                    
                    train = DataLoader(
                    train, shuffle=True, num_workers=args.data_loading_params['num_workers'], batch_size=args.data_loading_params['batch_size'], collate_fn=collate_func)
                    dev = DataLoader(
                    dev, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    
                    test_benign = DataLoader(
                        test_benign, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    
                    test_mixed = DataLoader(
                        test_iot23_mixed, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    test_malicious = DataLoader(
                        test_iot23_malicious, shuffle=False,  num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    
                    test_iotid20_benign = DataLoader(
                        test_iotid20_benign, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)
                    test_iotid20_mixed = DataLoader(
                        test_iotid20_mixed, shuffle=False, num_workers=args.data_loading_params['num_workers'], batch_size=1, collate_fn=collate_func)

                    
                self.tasker = tasker
                self.train = train
                self.dev = dev
                if not self.tasker.data.sequence:
                    self.test_benign = test_benign
                    
                    #self.test_iot23_benign = test_iot23_benign
                    self.test_mixed = test_iot23_mixed
                    self.test_malicious = test_iot23_malicious
                
                    self.test_iotid20_benign = test_iotid20_benign
                    self.test_iotid20_mixed = test_iotid20_mixed
                return
                
            self.tasker = tasker
            self.train = train
            self.dev = dev
            if not self.tasker.data.sequence:
                self.test_benign = test_benign
                self.test_mixed = test_mixed
                self.test_malicious = test_malicious
            else:
                self.test_iot23 = test_iot23

            self.test_traces = test_traces

            if not self.tasker.data.sequence:
                self.test_iotid20_benign = test_iotid20_benign
                self.test_iotid20_mixed = test_iotid20_mixed
            else:
                self.test_iotid20 = test_iotid20

    def anomaly_split(self, tasker, args):
        """_summary_

        Returns:
            _type_: _description_
        """
        # for each capture compute the max number of graphs
        dataset = tasker.data
        graph_list = dataset.graph_list
        train_graphs = OrderedDict()
        val_graphs = OrderedDict()
        for capture in graph_list.keys():
            train_graphs[capture] = OrderedDict()
            val_graphs[capture] = OrderedDict()
            for representation in graph_list[capture].keys():
                for graph_type in graph_list[capture].keys():
                    graph_number = len(
                        graph_list[capture][graph_type])
                    train_final_indx = int(graph_number*args.train_val)
                    train_graphs[capture][graph_type] = graph_list[capture][graph_type][:train_final_indx]
                    val_graphs[capture][graph_type] = graph_list[capture][graph_type][train_final_indx:]

        train_data = AnomalyDataset(graph_list=train_graphs)
        val_data = AnomalyDataset(graph_list=val_graphs)

        train_data, val_data, None


class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self, idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test=self.test, **self.kwargs)
        return t


class static_data_split(Dataset):
    def __init__(self, tasker, indexes, test):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx, test=self.test)


class AnomalyDataset(Dataset):
    def get_number(self, path):
        return int(path.split('/')[-1].split('.')[0].split('_')[-1])

    def __init__(self, tasker: nat.Anomaly_Detection_Tasker, path: str = None, split: str = 'train', representation: str = 'etdg_graph'):
        self.data_dict = None
        self.total_number_graphs = 0
        self.number_of_captures = 0
        self.indx_to_graph = OrderedDict()
        self.capture_start_end_indx = OrderedDict()
        self.tasker = tasker
        self.representation = representation
        self.mode = split

        if split == "train":
            self.data_dict_path = os.path.join(path, "train.json")
        elif split == "val":
            self.data_dict_path = os.path.join(path, "val.json")
        elif split == "test_mixed":
            self.data_dict_path = os.path.join(path, "test_mixed.json")
        elif split == "test_malicious":
            self.data_dict_path = os.path.join(path, "test_malicious.json")
        elif split == "test_benign":
            self.data_dict_path = os.path.join(path, "test_benign.json")
        elif split == "test_iot23":
            self.data_dict_path = [os.path.join(path, "test_benign.json"),
                                   os.path.join(path, "test_malicious.json"),
                                   os.path.join(path, "test_mixed.json")]
        elif split == "IoT_traces":
            self.data_dict_path = os.path.join(path, "test_benign.json")
        elif split == "IoTID20_benign":
            self.data_dict_path = os.path.join(path, "test_benign.json")
        elif split == "IoTID20_mixed":
            self.data_dict_path = os.path.join(path, "test_mixed.json")
        elif split == "IoTID20":
            self.data_dict_path = [os.path.join(path, "test_benign.json"),
                                   os.path.join(path, "test_mixed.json")]
            
        elif split == "test_benign_iot_traces":
            self.data_dict_path = os.path.join(path.replace('/split/', '/split_test/'), "test_benign.json")

        if not self.tasker.data.sequence or not isinstance(self.data_dict_path, list):
            with open(self.data_dict_path) as data_file:
                self.data_dict = json.load(data_file)

                """
                if split == 'test_mixed' or split == "train" or split == "val":
                    self.data_dict.pop('CTU-IoT-Malware-Capture-9-1', None)
                    self.data_dict.pop('CTU-IoT-Malware-Capture-1-1', None)
                    self.data_dict.pop('CTU-IoT-Malware-Capture-48-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-3-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-42-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-34-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-20-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-44-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-21-1', None)
                    # self.data_dict.pop('CTU-IoT-Malware-Capture-8-1', None)
                    # self.data_dict.pop('CTU-Honeypot-Capture-4-1', None)
                    # self.data_dict.pop('CTU-Honeypot-Capture-7-1', None)
                    # self.data_dict.pop('CTU-Honeypot-Capture-5-1', None)
                """
            self.number_of_captures = len(list(self.data_dict.keys()))

            # create a map between graph path and index
            graph_indx = 0
            self.total_number_graphs = 0
            for capture in self.data_dict.keys():
                # if "48-1" not in capture and "9-1" not in capture:
                # if "Honeypot" in capture:
                # print(f"Loading capture {capture}")
                self.capture_start_end_indx[capture] = OrderedDict()
                graph_cnt = 0
                for sequence_indx, sequence in enumerate(self.data_dict[capture]):
                    start_indx = graph_indx
                    for graph in sequence:
                        # if self.check_graph(data_dict_path=self.data_dict_path,
                        #                     graph_name=graph,
                        #                     capture_name=capture):
                        self.indx_to_graph[graph_indx] = (
                            capture, graph, sequence_indx)
                        graph_indx += 1
                        graph_cnt += 1
                    self.capture_start_end_indx[capture][sequence_indx] = [
                        start_indx, graph_indx-1]
                self.total_number_graphs += graph_cnt

        elif self.tasker.data.sequence and isinstance(self.data_dict_path, list):
            self.data_dict = dict()
            # fuse dict
            for indx, partition in enumerate(self.data_dict_path):
                partition_name = partition.split('/')[-1].split('.')[0]
                partition_to_graph = OrderedDict()
                with open(partition) as data_file:
                    self.data_dict_partition = json.load(data_file)
                # if partition_name == 'test_mixed':
                # self.data_dict.pop('CTU-IoT-Malware-Capture-3-1', None)
                for capture in self.data_dict_partition:
                    partition_to_graph[capture] = OrderedDict()
                    if self.data_dict.get(capture, None) is None:
                        self.data_dict[capture] = list()
                    for sequence in self.data_dict_partition[capture]:
                        for graph in sequence:
                            if "mixed" in partition_name:
                                partition = "mixed"
                            elif "benign" in partition_name:
                                partition = "full_benign"
                            elif "malicious" in partition_name:
                                partition = "full_malicious"
                            self.data_dict[capture].append(
                                f"{partition}/{graph}")
            # sort list
            graph_indx = 0
            for capture in self.data_dict:
                self.data_dict[capture].sort(key=self.get_number)
                self.total_number_graphs += len(self.data_dict[capture])
                start_indx = graph_indx
                self.capture_start_end_indx[capture] = OrderedDict()
                for _, graph in enumerate(self.data_dict[capture]):
                    self.indx_to_graph[graph_indx] = (
                        capture, graph, 0)
                    graph_indx += 1
                self.capture_start_end_indx[capture][0] = [
                    start_indx, graph_indx-1]

        print(f"---- {split}: loading completed ----")

    def __len__(self):
        return self.total_number_graphs

    def check_graph(self, data_dict_path, graph_name, capture_name):
        if self.mode == "train" or self.mode == "val" or self.mode == "test_mixed" or self.mode == "test_malicious" or self.mode == "test_benign":
            graph_base_folder = self.tasker.data.graph_base_folder
        elif self.mode == "IoT_traces":
            graph_base_folder = self.tasker.data.graph_base_iot_traces_folder
        else:
            graph_base_folder = self.tasker.data.graph_base_iot_id20_folder

        if "benign" in data_dict_path or "train" in data_dict_path or "val" in data_dict_path:
            graph_type = "full_benign"
        if "malicious" in data_dict_path:
            graph_type = "full_malicious"
        if "mixed" in data_dict_path:
            graph_type = "mixed"

        graph_path = os.path.join(
            graph_base_folder, capture_name, self.tasker.data.representation, graph_type, graph_name)
        if os.path.getsize(graph_path) / (1024 ** 3) > 0.009:
            # print(round(os.path.getsize(graph_path) / (1024 ** 3),  3))
            return False
        return True

    def __getitem__(self, idx):

        # map indx to capture
        capture = self.indx_to_graph[idx][0]
        sequence_indx = self.indx_to_graph[idx][2]

        start_indx, end_indx = self.capture_start_end_indx[capture][sequence_indx]

        if self.mode == "train":
            graph_type = "full_benign"
        elif self.mode == "val":
            graph_type = "full_benign"
        elif self.mode == "test_mixed":
            graph_type = "mixed"
        elif self.mode == "test_malicious":
            graph_type = "full_malicious"
        elif self.mode == "test_benign":
            graph_type = "full_benign"
        elif self.mode == "IoT_traces":
            graph_type = "full_benign"
        elif self.mode == "IoTID20_benign":
            graph_type = "full_benign"
        elif self.mode == "IoTID20_mixed":
            graph_type = "mixed"
        else:
            graph_type = None

        t = self.tasker.get_sample(
            idx=idx,
            start_indx=start_indx,
            end_indx=end_indx,
            graph_list=self.indx_to_graph,
            capture_name=capture,
            graph_type=graph_type,
            split=self.mode)
        return t
