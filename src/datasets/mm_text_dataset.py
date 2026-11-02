import numpy as np
import random, torch, joblib
from torch.utils.data import Dataset
from mmengine.registry import DATASETS, FUNCTIONS
import time
from torch.nn.utils.rnn import pad_sequence

FPS = 30
@DATASETS.register_module()
class Humanml3Dataset(Dataset):
    '''
    data format: 
    dict(
        motion_272: np.ndarray (T, 272)
        # text
        text: [[caption, start_time, end_time], ...], 
        # music
        music_feat: 
        # traj
        pivot_traj: 
    )
    '''

    def __init__(self, split_file='', window_size=-1, unit_length=4, mean_path='', std_path='', 
                 min_motion_length=60, max_motion_length=300, repeat_time=1, interval=1, add_traj=False):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.window_size = window_size  
        self.unit_length = unit_length
        self.repeat_time = repeat_time
        # self.condition_types = ['text', 'music', 'traj', 'video']
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
        self.add_traj = add_traj

    def __len__(self):
        return len(self.data_infos) * self.repeat_time
    
    def __getitem__(self, index):
        
        data_info = self.data_infos[index % len(self.data_infos)]
        # 1. load motion
        motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)

        # 2. load text condition
        text_data = ''
        has_text = False
        start_time, end_time = 0, 0
        if 'text' in data_info.keys():
            # text_data = random.choice(data_info['text'])
            text_data, start_time, end_time = random.choice(data_info['text'])
            has_text = True
            if end_time > 0: # start / end time 如果都是0的话, 表示不需要clip
                motion = motion[int(start_time * FPS):int(end_time * FPS)]
    
        # 3. clip motion
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
        return dict(
            motion=motion, 
            motion_length=motion_length, 

            mean=self.motion_mean,
            std=self.motion_std,
            
            text_data=text_data, 
        )

@DATASETS.register_module()
class MMDataset(Humanml3Dataset):
    def __getitem__(self, index):
        
        data_info = self.data_infos[index % len(self.data_infos)]
        # 1. load motion
        motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)

        # 2. load text condition
        text_data = random.choice(data_info['text'])
        # 3. clip motion
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
        return dict(
            motion=motion, 
            motion_length=motion_length, 

            mean=self.motion_mean,
            std=self.motion_std,
            
            text_data=text_data, 
        )


@FUNCTIONS.register_module()
def text_collate_fn(batch):
    motion = [item['motion'] for item in batch] # [motion: T, C]
    motion_length = torch.as_tensor([item['motion_length'] for item in batch])
    motion = pad_sequence(motion, batch_first=True)

    mean = batch[0]['mean']
    std = batch[0]['std']
    
    text_data = [item['text_data'] for item in batch]
    
    return dict(
        motion=motion, 
        motion_length=motion_length,

        mean=mean, 
        std=std, 
        
        text_data=text_data, 
    )