import numpy as np
import random, torch, joblib
from torch.utils.data import Dataset
from mmengine.registry import DATASETS, FUNCTIONS
import time, os
from ..utils.motion import recover_from_local_position_fast

FPS = 30
@DATASETS.register_module()
class AnyMoTextDataset(Dataset):
    '''
    data format: 
    dict(
        motion_272: np.ndarray (T, 272), 
        captions: [str]
    )
    '''
    def __init__(self, data_path_file='', unit_length=4, mean_path='', std_path='', 
                 min_motion_length=60, max_motion_length=300, interval=1):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
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
            caption = random.choice(data_info['captions'])

            # 2. clip motion
            T = motion.shape[0]
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
                text_data=caption,
                motion_length=motion_length, 
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data

@DATASETS.register_module()
class AnyMoTextDataset_v1(AnyMoTextDataset):
    '''
    加采样
    data format: 
    dict(
        motion_272: np.ndarray (T, 272), 
        captions: [str]
    )
    '''
    def __init__(self, sample_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.sample_ratio = sample_ratio

    def __len__(self):
        return int(len(self.data_paths) * sample_ratio)
    
    def __getitem__(self, index):
        cap_size = len(self.data_paths)
        size = int(cap_size * self.sample_ratio)
        begin = random.randint(0, cap_size - size)
        data_infos = joblib.load(self.caption_paths[begin + index])
        data = []
        for data_info in data_infos:
            # 1. load motion
            motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)
            caption = random.choice(data_info['captions'])

            # 2. clip motion
            T = motion.shape[0]
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
                text_data=caption,
                motion_length=motion_length, 
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data


@DATASETS.register_module()
class AnyMoTextTrajDataset(AnyMoTextDataset):
    def __init__(self, traj_mean_path='', traj_std_path='', **kwargs):
        super().__init__(**kwargs)
        self.traj_mean = torch.from_numpy(np.load(traj_mean_path).astype(np.float32))
        self.traj_std = torch.from_numpy(np.load(traj_std_path).astype(np.float32))
    def __getitem__(self, index):
        data_infos = joblib.load(self.data_paths[index])
        data = []
        for data_info in data_infos:
            # 1. load motion
            motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)
            caption = random.choice(data_info['captions'])

            traj_data = recover_from_local_position_fast(motion[None]).squeeze(0)[:, 0]

            # 2. clip motion
            T = motion.shape[0]
            motion_length = (T // self.unit_length) * self.unit_length
            if motion_length > self.max_motion_length:
                motion_length = self.max_motion_length
            idx = random.randint(0, T - motion_length)
            motion = motion[idx:idx+motion_length]
            traj_data = traj_data[idx:idx+motion_length]
            traj_data = (traj_data - self.traj_mean) / self.traj_std
            
            motion = (motion - self.motion_mean) / self.motion_std
            pad_motion = torch.zeros(self.max_motion_length - motion_length, motion.shape[1]).to(motion)
            motion = torch.cat([motion, pad_motion], dim=0)
            pad_traj = torch.zeros(self.max_motion_length - motion_length, traj_data.shape[1]).to(traj_data)
            traj_data = torch.cat([traj_data, pad_traj], dim=0)

            data.append(dict(
                motion=motion, 
                traj_data=traj_data,
                text_data=caption,
                motion_length=motion_length, 
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data

@FUNCTIONS.register_module()
def anymo_text_traj_collate_fn(batch):
    new_batch = []
    for item in batch:
        new_batch.extend(item)
    motion = [item['motion'] for item in new_batch] # [motion: T, C]
    traj_data = [item['traj_data'] for item in new_batch]
    traj_data = torch.stack(traj_data)
    text_data = [item['text_data'] for item in new_batch]
    motion_length = torch.as_tensor([item['motion_length'] for item in new_batch])
    motion = torch.stack(motion)
    motion = motion[:, :motion_length.max(), :]
    traj_data = traj_data[:, :motion_length.max()]

    mean = new_batch[0]['mean']
    std = new_batch[0]['std']
    return dict(
        motion=motion, 
        traj_data=traj_data,
        motion_length=motion_length,
        text_data=text_data,

        mean=mean, 
        std=std, 
    )

@FUNCTIONS.register_module()
def anymo_text_collate_fn(batch):
    new_batch = []
    for item in batch:
        new_batch.extend(item)
    motion = [item['motion'] for item in new_batch] # [motion: T, C]
    text_data = [item['text_data'] for item in new_batch]
    motion_length = torch.as_tensor([item['motion_length'] for item in new_batch])
    motion = torch.stack(motion)
    motion = motion[:, :motion_length.max(), :]

    mean = new_batch[0]['mean']
    std = new_batch[0]['std']
    return dict(
        motion=motion, 
        motion_length=motion_length,
        text_data=text_data,

        mean=mean, 
        std=std, 
    )
