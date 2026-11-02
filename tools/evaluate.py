"""Reproduce released text, trajectory and tokenizer metrics on one CUDA GPU."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def parse_args():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--assets-dir',type=Path,default=ROOT,help='Root containing data statistics, checkpoints and work_dirs')
 parser.add_argument('--data-root',type=Path,default=ROOT/'data',help='Directory containing the extracted omnihumo tree')
 parser.add_argument('--split-dir',type=Path,default=ROOT/'configs/data_paths')
 parser.add_argument('--checkpoint',type=Path)
 parser.add_argument('--config',type=Path,default=ROOT/'configs/inference/anymo_3b.py')
 parser.add_argument('--evaluator-config',type=Path)
 parser.add_argument('--evaluator-checkpoint',type=Path)
 parser.add_argument('--tokenizer-checkpoint',type=Path)
 parser.add_argument('--text-backbone',type=Path)
 parser.add_argument('--phases',nargs='+',choices=['reconstruction','text','text_traj'],default=['reconstruction','text','text_traj'])
 parser.add_argument('--seed',type=int,default=3407)
 parser.add_argument('--out',type=Path,default=ROOT/'outputs/full_evaluation')
 parser.add_argument('--max-batches',type=int,help='Smoke check only; outputs explicitly marked partial')
 parser.add_argument('--check-only',action='store_true',help='Validate inputs without importing Torch or starting CUDA')
 args=parser.parse_args()
 if args.max_batches is not None and args.max_batches<1:parser.error('--max-batches must be positive')
 for name,relative in [('checkpoint','work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth'),('evaluator_config','work_dirs/temos_50e/temos.py'),('evaluator_checkpoint','work_dirs/temos_50e/epoch_50.pth'),('tokenizer_checkpoint','work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth'),('text_backbone','checkpoints/distilbert-base-uncased')]:
  if getattr(args,name) is None:setattr(args,name,args.assets_dir/relative)
 for key,value in vars(args).items():
  if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
 return args

def validate(args):
 import json
 missing=[]
 for path in [args.checkpoint,args.config,args.evaluator_config,args.evaluator_checkpoint,args.tokenizer_checkpoint,args.text_backbone/'config.json',args.text_backbone/'tokenizer_config.json']+[args.assets_dir/'data'/(n+'.npy') for n in ['mean_272','std_272','traj_mean','traj_std']]:
  if not path.is_file():missing.append(str(path))
 names=['omnihumo_motion_caption_dynamic_test']
 if 'reconstruction' in args.phases:names=['omnihumo_vq_test','omnihumo_motion_caption_dynamic_test','omnihumo_motion_caption_static_test','dance_test','speak_test']
 checked=set()
 for name in names:
  path=args.split_dir/(name+'.txt')
  if not path.is_file():missing.append(str(path));continue
  for line in path.read_text().splitlines():
   value=line.strip().removeprefix('data/')
   if not value:continue
   p=Path(value)
   if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe test split path: '+value)
   target=args.data_root/p;checked.add(str(target))
   if not target.is_file():missing.append(str(target))
 result=dict(ready=not missing,test_files_checked=len(checked),missing_count=len(missing),missing_first20=missing[:20])
 print(json.dumps(result,indent=2))
 if missing:raise SystemExit('Missing evaluation assets; see docs/EVALUATION.md')

def main():
 args=parse_args();validate(args)
 if args.check_only:return
 import torch
 if not torch.cuda.is_available():raise SystemExit('A CUDA GPU is required')
 from src.evaluation.benchmark import run
 run(args)
if __name__=='__main__':main()
