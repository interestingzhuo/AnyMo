import numpy as np
import random, torch, joblib
from torch.utils.data import Dataset
from mmengine.registry import DATASETS, FUNCTIONS
import time, os
from torch.nn.utils.rnn import pad_sequence
from ..utils.motion import recover_from_local_position_fast

FPS = 30
Audio_FPS = 75
@DATASETS.register_module()
class AnyMoDataset(Dataset):
    '''
    data format: 
    dict(
        motion_272: np.ndarray (T, 272), 
        captions: [str]
    )
    '''
    def __init__(self, data_path_file='', unit_length=4, mean_path='', 
                 std_path='', audio_unit_length=8, min_motion_length=60, 
                 traj_mean_path='', traj_std_path='', max_motion_length=300, interval=1):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.audio_unit_length = audio_unit_length

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
        self.traj_mean = torch.from_numpy(np.load(traj_mean_path).astype(np.float32))
        self.traj_std = torch.from_numpy(np.load(traj_std_path).astype(np.float32))

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        data_infos = joblib.load(self.data_paths[index])
        data = []
        for data_info in data_infos:
            # 1. load motion
            motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)
            caption = random.choice(data_info['captions'])
            audio_feat = torch.as_tensor(data_info['audio_feat'], dtype=torch.float32)

            # traj_data = torch.as_tensor(data_info['root_traj'], dtype=torch.float32)
            traj_data = recover_from_local_position_fast(motion[None]).squeeze(0)[:, 0]

            # 有一些比较长的音频，这部分不用caption标注
            if audio_feat.shape[0] // Audio_FPS > 15:
                caption = ''

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
            
            # 3. clip audio
            audio_length = motion_length / FPS * Audio_FPS
            audio_length = int((audio_length // self.audio_unit_length) * self.audio_unit_length)
            audio_idx = int(idx / FPS * Audio_FPS)
            audio_feat = audio_feat[audio_idx:audio_idx+audio_length]

            data.append(dict(
                motion=motion, 
                traj_data=traj_data,
                text_data=caption,
                audio_data=audio_feat,
                motion_length=motion_length, 
                # audio_length=audio_length,
                audio_length=audio_feat.shape[0],
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data

@DATASETS.register_module()
class AnyMoDataset_v1(AnyMoDataset):
    # 允许数据缺失
    def __getitem__(self, index):
        data_infos = joblib.load(self.data_paths[index])
        data = []
        for data_info in data_infos:
            # 1. load motion
            motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)
            m_path = data_info['motion_272_path']
            caption = random.choice(data_info['captions'])
            has_audio = 'audio_feat' in data_info
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

            # 3. clip audio
            if has_audio:
                # 有一些比较长的音频，这部分不用caption标注
                audio_feat = torch.as_tensor(data_info['audio_feat'], dtype=torch.float32)
                if audio_feat.shape[0] // Audio_FPS > 15:
                    caption = ''
                audio_length = motion_length / FPS * Audio_FPS
                audio_length = int((audio_length // self.audio_unit_length) * self.audio_unit_length)
                audio_idx = int(idx / FPS * Audio_FPS)
                audio_feat = audio_feat[audio_idx:audio_idx+audio_length]

                beat_path = os.path.join("data/omnihumo/motion_caption_audio_beat", data_info['motion_272_path'])
                # print(np.load(beat_path).shape, motion_length)
                audio_beat = torch.from_numpy(np.load(beat_path)[0, :motion_length])
            else:
                audio_feat = torch.zeros((0, 512)).float()
                audio_length = 0
                audio_beat = torch.zeros((0, )).float()
            
            data.append(dict(
                motion=motion, 
                m_path=m_path, 
                traj_data=traj_data,
                text_data=caption,
                audio_data=audio_feat,
                motion_length=motion_length, 
                audio_length=audio_feat.shape[0],
                audio_beat=audio_beat, 
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data

@DATASETS.register_module()
class AnyMoDataset_v2(Dataset):
    # 对 text 采样 + 允许数据缺失
    def __init__(self, audio_path_file='', caption_path_file='', unit_length=4, mean_path='', 
                 std_path='', audio_unit_length=8, min_motion_length=60, caption_sample_ratio=0.3, 
                 traj_mean_path='', traj_std_path='', max_motion_length=300, interval=1):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.audio_unit_length = audio_unit_length

        audio_paths = []
        for path in audio_path_file:
            with open(path, 'r') as f:
                audio_paths.extend([line.strip() for line in f.readlines()])

        with open(caption_path_file, 'r') as f:
            caption_paths = [line.strip() for line in f.readlines()]

        self.audio_paths = audio_paths[::interval]
        self.caption_paths = caption_paths[::interval]
        self.caption_sample_ratio = caption_sample_ratio # 对caption采样率

        self.motion_mean = torch.from_numpy(np.load(mean_path).astype(np.float32))
        self.motion_std = torch.from_numpy(np.load(std_path).astype(np.float32))
        self.traj_mean = torch.from_numpy(np.load(traj_mean_path).astype(np.float32))
        self.traj_std = torch.from_numpy(np.load(traj_std_path).astype(np.float32))

    def __len__(self):
        return len(self.audio_paths) + int(len(self.caption_paths) * self.caption_sample_ratio)
    
    def __getitem__(self, index):
        if index < len(self.audio_paths):
            data_infos = joblib.load(self.audio_paths[index])
        else:
            index = index - len(self.audio_paths)
            cap_size = len(self.caption_paths)
            size = int(cap_size * self.caption_sample_ratio)
            begin = random.randint(0, cap_size - size)
            data_infos = joblib.load(self.caption_paths[begin + index])

        data = []
        for data_info in data_infos:
            # 1. load motion
            motion = torch.as_tensor(data_info['motion_272'], dtype=torch.float32)
            caption = random.choice(data_info['captions'])
            has_audio = 'audio_feat' in data_info
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

            # 3. clip audio
            if has_audio:
                # 有一些比较长的音频，这部分不用caption标注
                audio_feat = torch.as_tensor(data_info['audio_feat'], dtype=torch.float32)
                if audio_feat.shape[0] // Audio_FPS > 15:
                    caption = ''
                audio_length = motion_length / FPS * Audio_FPS
                audio_length = int((audio_length // self.audio_unit_length) * self.audio_unit_length)
                audio_idx = int(idx / FPS * Audio_FPS)
                audio_feat = audio_feat[audio_idx:audio_idx+audio_length]
            else:
                audio_feat = torch.zeros((0, 512)).float()
                audio_length = 0


            data.append(dict(
                motion=motion, 
                traj_data=traj_data,
                text_data=caption,
                audio_data=audio_feat,
                motion_length=motion_length, 
                audio_length=audio_feat.shape[0],
                mean=self.motion_mean,
                std=self.motion_std,
            ))

        return data

@FUNCTIONS.register_module()
def anymo_collate_fn(batch):
    new_batch = []
    for item in batch:
        new_batch.extend(item)
    motion = [item['motion'] for item in new_batch] # [motion: T, C]
    motion = torch.stack(motion)

    traj_data = [item['traj_data'] for item in new_batch]
    traj_data = torch.stack(traj_data)

    motion_length = torch.as_tensor([item['motion_length'] for item in new_batch])
    audio_length = torch.as_tensor([item['audio_length'] for item in new_batch])
    text_data = [item['text_data'] for item in new_batch]
    motion = motion[:, :motion_length.max()]
    traj_data = traj_data[:, :motion_length.max()]

    mean = new_batch[0]['mean']
    std = new_batch[0]['std']

    audio_data = [item['audio_data'] for item in new_batch]
    audio_data = pad_sequence(audio_data, batch_first=True)
    if 'audio_beat' in new_batch[0]:
        audio_beat = [item['audio_beat'] for item in new_batch]
        audio_beat = pad_sequence(audio_beat, batch_first=True)
    else:
        audio_beat = None
    # print(motion.shape, traj_data.shape, audio_data.shape, len(text_data))
    return dict(
        motion=motion, 
        traj_data=traj_data,
        audio_data=audio_data,
        motion_length=motion_length,
        audio_length=audio_length,
        text_data=text_data,
        audio_beat=audio_beat, 

        mean=mean, 
        std=std, 
    )


@FUNCTIONS.register_module()
def anymo_collate_fn_v1(batch):
    new_batch = []
    for item in batch:
        new_batch.extend(item)
    motion = [item['motion'] for item in new_batch] # [motion: T, C]
    m_path = [item['m_path'] for item in new_batch] # [motion: T, C]
    motion = torch.stack(motion)

    traj_data = [item['traj_data'] for item in new_batch]
    traj_data = torch.stack(traj_data)

    motion_length = torch.as_tensor([item['motion_length'] for item in new_batch])
    audio_length = torch.as_tensor([item['audio_length'] for item in new_batch])
    text_data = [item['text_data'] for item in new_batch]
    motion = motion[:, :motion_length.max()]
    traj_data = traj_data[:, :motion_length.max()]

    mean = new_batch[0]['mean']
    std = new_batch[0]['std']

    audio_data = [item['audio_data'] for item in new_batch]
    audio_data = pad_sequence(audio_data, batch_first=True)
    if 'audio_beat' in new_batch[0]:
        audio_beat = [item['audio_beat'] for item in new_batch]
        audio_beat = pad_sequence(audio_beat, batch_first=True)
    else:
        audio_beat = None
    # print(motion.shape, traj_data.shape, audio_data.shape, len(text_data))
    return dict(
        motion=motion, 
        traj_data=traj_data,
        audio_data=audio_data,
        motion_length=motion_length,
        audio_length=audio_length,
        text_data=text_data,
        audio_beat=audio_beat, 
        m_path=m_path, 
        mean=mean, 
        std=std, 
    )
