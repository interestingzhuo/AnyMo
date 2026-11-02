import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
src.register_all_modules()
from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.metrics.t2m_metric import calculate_frechet_distance, calculate_activation_statistics
from src.datasets.anymo_text_dataset import anymo_text_collate_fn
import numpy as np

cfg = Config.fromfile('configs/temos.py')
model = MODELS.build(cfg.model)

ckpt = torch.load('work_dirs/temos_50e/epoch_50.pth', weights_only=False, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.cuda().eval()

mean_path='data/mean_272.npy'
std_path='data/std_272.npy'
train_dataset = dict(
    type='AnyMoTextDataset',
    data_path_file=[
        # 'configs/data_paths/omnihumo_motion_caption_dynamic.txt', 
        # 'configs/data_paths/omnihumo_motion_caption_static.txt'
        # "configs/data_paths/omnihumo_motion_caption_speak.txt", 
        "configs/data_paths/omnihumo_motion_caption_dance.txt"
    ], 
    unit_length=4, 
    mean_path=mean_path,
    std_path=std_path,
    min_motion_length=60, 
    max_motion_length=300,
    interval=1, 
)

test_dataset = dict(
    type='AnyMoTextDataset',
    data_path_file='configs/data_paths/dance_test.txt', 
    unit_length=4, 
    mean_path=mean_path,
    std_path=std_path,
    min_motion_length=60, 
    max_motion_length=300,
    interval=1, 
)

train_dataset = DATASETS.build(train_dataset)
test_dataset = DATASETS.build(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=anymo_text_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=anymo_text_collate_fn)

train_latents = []
for data in tqdm(train_dataloader):
    motion = data['motion'].cuda()
    m_len = data['motion_length'].cuda()
    pad_m = torch.zeros((motion.shape[0], 300 - m_len.max(), 272)).to(motion)
    motion = torch.cat([motion, pad_m], dim=1)
    with torch.no_grad():
        latents = model.motionencoder(motion.cuda(), m_len.cuda()).loc
    train_latents.append(latents.cpu().numpy())

test_latents = []
for data in tqdm(test_dataloader):
    motion = data['motion'].cuda()
    m_len = data['motion_length'].cuda()
    pad_m = torch.zeros((motion.shape[0], 300 - m_len.max(), 272)).to(motion)
    motion = torch.cat([motion, pad_m], dim=1)
    with torch.no_grad():
        latents = model.motionencoder(motion.cuda(), m_len.cuda()).loc
    test_latents.append(latents.cpu().numpy())

train_embd = np.concatenate(train_latents, axis=0)
test_embd = np.concatenate(test_latents, axis=0)

all_embd = np.concatenate([train_embd, test_embd], axis=0)
np.save('all_gt_dance_embd.npy', all_embd)

gt_mu, gt_cov = calculate_activation_statistics(train_embd)
mu, cov = calculate_activation_statistics(test_embd)

fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
print(fid)
