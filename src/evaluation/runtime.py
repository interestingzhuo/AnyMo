"""Strict-load released model and official metric encoder subsets."""
from pathlib import Path
import importlib,types,sys
from unittest.mock import patch
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from transformers import AutoConfig,AutoModel
CODE=Path(__file__).resolve().parents[2]
from src.inference import load_model,load_stats
from src.utils.motion import recover_from_local_position_fast as recover

def load_runtime(args):
 model,cfg=load_model(args.checkpoint,args.config,args.assets_dir,['text'])
 model.inference_timesteps=6;model.topk_filter_thres=.9
 stats=load_stats(args.assets_dir,True)
 namespace=types.ModuleType('src.models.evaluator');namespace.__path__=[str(CODE/'src/models/evaluator')];sys.modules['src.models.evaluator']=namespace
 for name in ['motion_actor','distillbert_actor']:importlib.import_module('src.models.evaluator.'+name)
 config=Config.fromfile(str(args.evaluator_config))
 state=torch.load(args.evaluator_checkpoint,map_location='cpu',mmap=True,weights_only=False)['state_dict']
 motion=MODELS.build(config.model.motion_encoder_cfg)
 textcfg=config.model.text_encoder_cfg.copy();textcfg.modelpath=str(args.text_backbone)
 with patch('src.models.evaluator.distillbert_actor.AutoModel.from_pretrained',side_effect=lambda p,**kw:AutoModel.from_config(AutoConfig.from_pretrained(p,local_files_only=True))):text=MODELS.build(textcfg)
 for prefix,encoder in [('motionencoder.',motion),('textencoder.',text)]:
  subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
  encoder.load_state_dict(subset,strict=True);encoder.eval().requires_grad_(False).cuda()
 print('PAPER_STRICT_GENERATOR_AND_ENCODERS_PASSED',flush=True)
 return model,stats,motion,text

def official_metrics():
 ns=types.ModuleType('src.metrics');ns.__path__=[str(CODE/'src/metrics')];sys.modules['src.metrics']=ns
 return importlib.import_module('src.metrics.t2m_metric')
