import numpy as np
import random, torch, joblib
from torch.utils.data import Dataset
from mmengine.registry import DATASETS, FUNCTIONS
import time, os

FPS = 30
@DATASETS.register_module()
class AnyMoMotionDataset(Dataset):
    '''
    data format: 
    dict(
        motion_272: np.ndarray (T, 272)
    )
    '''

    def __init__(self, data_path_file='', window_size=-1, unit_length=4, mean_path='', std_path='', 
                 min_motion_length=60, max_motion_length=300, interval=1):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.window_size = window_size  
        self.unit_length = unit_length

        if isinstance(data_path_file, list):
            self.data_paths = []
            for path in data_path_file:
                with open(path, 'r') as f:
                    self.data_paths.extend([line.strip() for line in f.readlines()])

        elif isinstance(data_path_file, str):
            with open(data_path_file, 'r') as f:
                self.data_paths = [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
            
        self.data_paths = self.data_paths[::interval]
        self.motion_mean = torch.from_numpy(np.load(mean_path).astype(np.float32))
        self.motion_std = torch.from_numpy(np.load(std_path).astype(np.float32))

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        data_infos = joblib.load(self.data_paths[index])
        data = []
        for data_info in data_infos:
            # 1. load motion
            motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)

            # 2. clip motion
            T = motion.shape[0]
            if self.window_size > 0:
                assert T >= self.window_size
                idx = 0 if T == self.window_size else random.randint(0, T - self.window_size)
                motion = motion[idx:idx+self.window_size]
                motion_length = self.window_size
            else:
                motion_length = (T // self.unit_length) * self.unit_length
                if motion_length > self.max_motion_length:
                    motion_length = self.max_motion_length
                idx = random.randint(0, T - motion_length)
                motion = motion[idx:idx+motion_length]
            
            motion = (motion - self.motion_mean) / self.motion_std
            pad_motion = torch.zeros(self.max_motion_length - motion_length, motion.shape[1]).to(motion)
            motion = torch.cat([motion, pad_motion], dim=0)
            data.append(dict(
                motion=motion, 
                motion_length=motion_length, 
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data

@FUNCTIONS.register_module()
def anymo_motion_collate_fn(batch):
    new_batch = []
    for item in batch:
        new_batch.extend(item)
    motion = [item['motion'] for item in new_batch] # [motion: T, C]
    motion_length = torch.as_tensor([item['motion_length'] for item in new_batch])
    motion = torch.stack(motion)

    mean = new_batch[0]['mean']
    std = new_batch[0]['std']
    return dict(
        motion=motion, 
        motion_length=motion_length,

        mean=mean, 
        std=std, 
    )


@DATASETS.register_module()
class MotionDataset(Dataset):
    '''
    data format: 
    dict(
        motion_272: np.ndarray (T, 272)
    )
    '''

    def __init__(self, split_file='', window_size=-1, unit_length=4, mean_path='', std_path='', 
                 min_motion_length=60, max_motion_length=300, repeat_time=1, interval=1):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.window_size = window_size  
        self.unit_length = unit_length
        self.repeat_time = repeat_time
        st = time.time()
        print('load data...')
        data_infos = joblib.load(split_file)
        print(f'load data with {time.time() - st}s')
        self.data_infos = []
        for data_info in data_infos:
            if data_info['motion_272'].shape[0] > self.min_motion_length:
                self.data_infos.append(data_info)
        self.data_infos = self.data_infos[::interval]
        self.motion_mean = torch.from_numpy(np.load(mean_path).astype(np.float32))
        self.motion_std = torch.from_numpy(np.load(std_path).astype(np.float32))

    def __len__(self):
        return len(self.data_infos) * self.repeat_time
    
    def __getitem__(self, index):
        
        data_info = self.data_infos[index % len(self.data_infos)]
        # 1. load motion
        motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)

        # 2. clip motion
        T = motion.shape[0]
        if self.window_size > 0:
            assert T >= self.window_size
            idx = 0 if T == self.window_size else random.randint(0, T - self.window_size)
            motion = motion[idx:idx+self.window_size]
            motion_length = self.window_size
        else:
            motion_length = (T // self.unit_length) * self.unit_length
            if motion_length > self.max_motion_length:
                motion_length = self.max_motion_length
            idx = random.randint(0, T - motion_length)
            motion = motion[idx:idx+motion_length]
        
        motion = (motion - self.motion_mean) / self.motion_std
        pad_motion = torch.zeros(self.max_motion_length - motion_length, motion.shape[1]).to(motion)
        motion = torch.cat([motion, pad_motion], dim=0)

        return dict(
            motion=motion, 
            motion_length=motion_length, 
            mean=self.motion_mean,
            std=self.motion_std,
        )

@FUNCTIONS.register_module()
def motion_collate_fn(batch):
    motion = [item['motion'] for item in batch] # [motion: T, C]
    motion_length = torch.as_tensor([item['motion_length'] for item in batch])
    motion = torch.stack(motion)

    mean = batch[0]['mean']
    std = batch[0]['std']
    return dict(
        motion=motion, 
        motion_length=motion_length,

        mean=mean, 
        std=std, 
    )
