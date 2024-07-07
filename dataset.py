import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import json
import math
from torch.utils.data import Dataset, ConcatDataset

class IMUArkitDataset_single(Dataset):
    def __init__(
            self, 
            datapath, 
            window_size = 120, 
            imu_desired_cols = ["Acceleration_X","Acceleration_Y","Acceleration_Z","Quaterion_0","Quaterion_1","Quaterion_2","Quaterion_3"],
            arkit_to_ict_file = 'configs/arkit_to_ict.json',
            mode='train', 
            overlap=60,
        ):
        
        self.datapath = datapath
        self.window_size = window_size
        self.mode = mode
        self.overlap = overlap if mode == 'test' else self.window_size - 1

        imu_idx_list = [0, 6, 7, 5, 1, 2, 13, 10, 18, 17, 20]

        imu_folder_path = os.path.join(datapath, 'IMU_data/calib')
        bs_file_path = os.path.join(datapath, 'Arkit/arkit_params.csv')

        imu_file_path_list = [os.path.join(imu_folder_path, '{}.csv'.format(str(imu_idx).zfill(3))) for imu_idx in imu_idx_list]
        
        imu_datas = []
        for imu_file_path in imu_file_path_list:
            imu_data = pd.read_csv(imu_file_path)[imu_desired_cols]
            imu_datas.append(imu_data)
        imu_datas = pd.concat(imu_datas, axis=1, ignore_index=False).to_numpy()

        if os.path.isfile(bs_file_path):
            bs_data = pd.read_csv(bs_file_path)
            with open(arkit_to_ict_file, 'r') as f:
                llf_mapping = pd.read_json(f)
            arkit_desired_cols = llf_mapping[0].tolist()
            bs_data = bs_data[arkit_desired_cols].to_numpy()
        else:
            print("Warning: No bs_file")
            bs_data = np.zeros()
        
        self.imu_datas = imu_datas.astype(np.float32)
        self.bs_data = bs_data.astype(np.float32)

        assert len(self.imu_datas) == len(self.bs_data)

        self.data_seq_len = len(self.bs_data)

    def __len__(self):
        return math.ceil((len(self.bs_data) - self.window_size) / (self.window_size - self.overlap))+1

    def __getitem__(self, idx):
        idx = idx * (self.window_size - self.overlap)
        if idx + self.window_size > self.data_seq_len:
            idx = self.data_seq_len - self.window_size 
        

        imu = self.imu_datas[idx:idx+self.window_size]
        bs = self.bs_data[idx:idx+self.window_size]
        actual_seq_len = len(imu)

        return {'imu': imu, 
                'bs': bs, 
                'seq_len': actual_seq_len,
                'data_seq_len': self.data_seq_len,
                'idxs': np.array(list(range(idx, idx+self.window_size, 1))),
                'datapath': self.datapath,
                }

class IMUArkitDataset(Dataset):
    def __init__(
            self, 
            datapath_list, 
            window_size = 120, 
            imu_desired_cols = ["Acceleration_X","Acceleration_Y","Acceleration_Z","Quaterion_0","Quaterion_1","Quaterion_2","Quaterion_3"],
            arkit_to_ict_file = 'configs/arkit_to_ict.json',
            mode='train', 
            overlap=0,
        ):
        self.datapath_list = datapath_list
        self.window_size = window_size
        self.overlap = overlap
        self.mode = mode
        
        self.datasets = []

        for datapath in datapath_list:
            print(datapath)
            self.datasets.append(IMUArkitDataset_single(
                datapath, 
                window_size, 
                imu_desired_cols = imu_desired_cols,
                arkit_to_ict_file = arkit_to_ict_file,
                mode=mode, 
                overlap=overlap,
            ))
        
        
        self.combined_dataset = ConcatDataset(self.datasets)


    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]