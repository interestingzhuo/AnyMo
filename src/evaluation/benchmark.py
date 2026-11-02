"""All released test records, original datasets/crops/collators, seed3407, FP32.
Resumable batch artifacts preserve the CUDA RNG and exact dataset batch digest.
"""
import os,json,time,random,hashlib,argparse,functools,subprocess,sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .runtime import CODE,load_runtime,official_metrics,recover
from mmengine.registry import MODELS
from mmengine.dataset import worker_init_fn,DefaultSampler
from src.datasets.anymo_text_dataset import AnyMoTextDataset,AnyMoTextTrajDataset,anymo_text_collate_fn,anymo_text_traj_collate_fn
from src.datasets.anymo_motion_dataset import AnyMoMotionDataset,anymo_motion_collate_fn
OUT=None
SEED=3407
ARGS=None
spec=None

def configure(args):
 global OUT,SEED,ARGS,spec
 ARGS=args;OUT=args.out;OUT.mkdir(parents=True,exist_ok=True);SEED=args.seed
 names=['omnihumo_vq_test','omnihumo_motion_caption_dynamic_test','omnihumo_motion_caption_static_test','dance_test','speak_test']
 splits={};hashes={}
 for name in names:
  p=args.split_dir/(name+'.txt');values=p.read_text().splitlines()
  splits[name]=[v.strip().removeprefix('data/') for v in values if v.strip()]
  hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 spec=dict(splits=splits,split_hashes=hashes)
 for values in splits.values():
  for value in values:
   p=Path(value)
   if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe split path: '+value)
 # Reject stale batches when model, metrics, source, paths or protocol change.
 def sha(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  return h.hexdigest()
 relevant=[args.checkpoint,args.config,args.evaluator_checkpoint,args.evaluator_config,
           args.tokenizer_checkpoint]+list(sorted((args.assets_dir/'data').glob('*.npy')))+list(sorted((args.text_backbone).glob('*')))
 source=hashlib.sha256()
 for p in sorted((CODE/'src').rglob('*.py')):source.update(p.relative_to(CODE).as_posix().encode());source.update(p.read_bytes())
 signature=dict(seed=SEED,max_batches=args.max_batches,split_hashes=hashes,data_root=str(args.data_root),
  artifacts={str(p):sha(p) for p in relevant if p.is_file()},source_sha256=source.hexdigest(),
  dependencies={name:version(name) for name in ['torch','numpy','scipy','transformers','mmengine','joblib']},
  dataset_manifest_sha256=sha(args.data_root/'evaluation_manifest.json') if (args.data_root/'evaluation_manifest.json').is_file() else None)
 target=OUT/'run_signature.json'
 if target.exists():
  if json.loads(target.read_text())!=signature:raise ValueError('Output directory belongs to a different model/protocol/source; choose a new --out')
 elif any(OUT.glob('*/*.npz')):raise ValueError('Cached batches have no run signature; choose a new --out')
 atomic_json(target,signature)

def atomic_json(path,value):
 temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False

def data_loader(kind):
 names=['omnihumo_vq_test','omnihumo_motion_caption_dynamic_test','omnihumo_motion_caption_static_test','dance_test','speak_test'] if kind=='reconstruction' else ['omnihumo_motion_caption_dynamic_test']
 paths=[]
 for name in names:
  p=OUT/(name+'.txt');p.write_text('\n'.join(str(ARGS.data_root/v) for v in spec['splits'][name])+'\n');paths.append(str(p))
 common=dict(data_path_file=paths,unit_length=4,mean_path=str(ARGS.assets_dir/'data/mean_272.npy'),std_path=str(ARGS.assets_dir/'data/std_272.npy'),min_motion_length=60,max_motion_length=300)
 if kind=='reconstruction':
  dataset=AnyMoMotionDataset(**common,window_size=-1);collate=anymo_motion_collate_fn;workers=8;batch=4
 else:
  cls=AnyMoTextTrajDataset if kind=='text_traj' else AnyMoTextDataset
  if kind=='text_traj':common.update(traj_mean_path=str(ARGS.assets_dir/'data/traj_mean.npy'),traj_std_path=str(ARGS.assets_dir/'data/traj_std.npy'))
  dataset=cls(**common);collate=anymo_text_traj_collate_fn if kind=='text_traj' else anymo_text_collate_fn;workers=4;batch=2
 sampler=DefaultSampler(dataset,shuffle=(kind!='reconstruction'),seed=SEED,round_up=True)
 loader=DataLoader(dataset,batch_size=batch,sampler=sampler,num_workers=workers,persistent_workers=False,pin_memory=True,collate_fn=collate,
                   worker_init_fn=functools.partial(worker_init_fn,num_workers=workers,rank=0,seed=SEED))
 return loader

def digest(data):
 h=hashlib.sha256()
 for name in ['motion','motion_length','traj_data']:
  if name in data:h.update(data[name].numpy().tobytes())
 if 'text_data' in data:h.update(json.dumps(data['text_data'],ensure_ascii=False).encode())
 return h.hexdigest()
def save_batch(path,**arrays):
 temp=path.with_suffix('.partial')
 with temp.open('wb') as f:np.savez_compressed(f,**arrays)
 temp.replace(path)
def progress(phase,index,total,count,started,computed):
 elapsed=time.monotonic()-started;eta=(total-index-1)*elapsed/max(computed,1)
 value=dict(status='running',phase=phase,completed_batches=index+1,total_batches=total,samples=count,elapsed_seconds=elapsed,eta_seconds=eta)
 atomic_json(OUT/'progress.json',value)
 if index%5==0 or index+1==total:print('PAPER_PROGRESS',json.dumps(value),flush=True)

def reconstruction(model,stats,metrics):
 target=OUT/'reconstruction.json'
 if target.exists():return json.loads(target.read_text())
 seed();loader=data_loader('reconstruction');folder=OUT/'reconstruction_batches';folder.mkdir(exist_ok=True)
 # The published standalone tokenizer is the evaluation checkpoint, not a replacement tokenizer.
 state=torch.load(ARGS.tokenizer_checkpoint,map_location='cpu',mmap=True,weights_only=False)['state_dict']
 embedded=model.vqvae.state_dict()
 equal=set(state)==set(embedded) and all(torch.equal(v,embedded[k].cpu()) for k,v in state.items())
 assert equal,'Standalone tokenizer differs from the embedded generator tokenizer'
 print('PAPER_STANDALONE_TOKENIZER_EQUAL',equal,flush=True)
 total_sum=0.;total_count=0;frame_sum=0.;frame_count=0;validation=None;started=time.monotonic();computed=0
 with torch.no_grad():
  for i,data in enumerate(loader):
   if ARGS.max_batches and i>=ARGS.max_batches:break
   path=folder/f'{i:06d}.npz';sha=digest(data)
   if path.exists():
    with np.load(path) as saved:
     assert str(saved['input_sha256'])==sha
     errors=saved['mpjpe_mm'].copy();nframes=saved['frames'].copy()
   else:
    gt=data['motion'].cuda(non_blocking=True);lengths=data['motion_length'].cuda(non_blocking=True)
    pred,_=model.vqvae(gt,mode='predict',motion_length=lengths);assert torch.isfinite(pred).all()
    p=recover(pred*stats['std_272']+stats['mean_272'],22);g=recover(gt*stats['std_272']+stats['mean_272'],22)
    error=((p-p[:,:,:1])-(g-g[:,:,:1])).norm(dim=-1).mean(-1)
    mask=torch.arange(error.shape[1],device='cuda')[None]<lengths[:,None]
    errors=((error*mask).sum(1)/lengths*1000).cpu().numpy();nframes=lengths.cpu().numpy()
    if validation is None:
     # Compare the GPU implementation against the actual released NumPy recovery and metric.
     from src.metrics.recons_metric import ReconsMetric,recover_from_local_position
     exact=[]
     for j,length in enumerate(nframes):
      a=recover_from_local_position((pred[j,:length]*stats['std_272']+stats['mean_272']).cpu().numpy(),22)
      b=recover_from_local_position((gt[j,:length]*stats['std_272']+stats['mean_272']).cpu().numpy(),22)
      exact.append(float(ReconsMetric().calc_mpjpe(torch.from_numpy(b).float()*1000,torch.from_numpy(a).float()*1000).mean()))
     validation=float(np.max(np.abs(errors-np.array(exact))));assert validation<.001,validation
     atomic_json(OUT/'reconstruction_metric_equivalence.json',dict(max_mpjpe_difference_mm=validation,passed=True))
     print('PAPER_NUMPY_MPJPE_EQUIVALENCE_PASSED',validation,flush=True)
    save_batch(path,mpjpe_mm=errors,frames=nframes,input_sha256=sha);computed+=1
   total_sum+=float(errors.astype(np.float64).sum());total_count+=len(errors)
   frame_sum+=float((errors.astype(np.float64)*nframes).sum());frame_count+=int(nframes.sum())
   progress('reconstruction',i,min(len(loader),ARGS.max_batches or len(loader)),total_count,started,computed)
 result=dict(status='partial' if ARGS.max_batches else 'complete',samples=total_count,files=len(loader.dataset),mpjpe_mm=total_sum/total_count,
             frame_weighted_mpjpe_mm=frame_sum/frame_count,standalone_equals_embedded=equal,aggregation='per-motion mean, matching ReconsMetric',seed=SEED)
 atomic_json(target,result);print('PAPER_RECONSTRUCTION_COMPLETE',result,flush=True)
 return result

def gen_metrics(folder,metrics,expected_batches):
 paths=sorted(folder.glob('*.npz'));assert len(paths)==expected_batches
 text=[];gt=[];pred=[];geos=[];rs=np.zeros(3);rreal=np.zeros(3);rm=np.zeros(3);match=0.;matchreal=0.;count=0;tokens=0;outside=0;sizes={}
 for path in paths:
  with np.load(path) as x:
   t=x['text'];g=x['gt'];p=x['pred'];n=len(t);sizes[n]=sizes.get(n,0)+1
   a,b=metrics.calculate_R_precision(t,p,3,True);c,d=metrics.calculate_R_precision(t,g,3,True);e,_=metrics.calculate_R_precision(p,t,3,True)
   rs+=a;rreal+=c;rm+=e;match+=float(b);matchreal+=float(d);count+=n
   text.append(t.copy());gt.append(g.copy());pred.append(p.copy());geos.append(x['trajectory'].copy());tokens+=int(x['tokens']);outside+=int(x['outside'])
 g=np.concatenate(gt);p=np.concatenate(pred);geo=np.concatenate(geos)
 mu_g,cov_g=metrics.calculate_activation_statistics(g);mu_p,cov_p=metrics.calculate_activation_statistics(p)
 fid=float(metrics.calculate_frechet_distance(mu_g,cov_g,mu_p,cov_p))
 convergence=[]
 for size in [512,1024,2048,4096,8192,16384,32768,count]:
  if size>count:continue
  a,ac=metrics.calculate_activation_statistics(g[:size]);b,bc=metrics.calculate_activation_statistics(p[:size])
  convergence.append(dict(samples=size,fid=float(metrics.calculate_frechet_distance(a,ac,b,bc))))
 pairs=min(300,count-1)
 np.random.seed(SEED);divreal=float(metrics.calculate_diversity(g,pairs));div=float(metrics.calculate_diversity(p,pairs))
 result=dict(status='complete',sample_count=count,batch_count=len(paths),candidate_group_sizes=sizes,
             R_text_to_motion=(rs/count).tolist(),R_motion_to_text=(rm/count).tolist(),R_real_text_to_motion=(rreal/count).tolist(),
             fid=fid,mmdist=match/count,mmdist_real=matchreal/count,diversity=div,diversity_real=divreal,
             trajectory=dict(traj_fail_20cm=float(geo[:,0].mean()),traj_fail_50cm=float(geo[:,1].mean()),location_fail_20cm=float(geo[:,2].mean()),location_fail_50cm=float(geo[:,3].mean()),ade_m=float(geo[:,4].mean()),fde_m=float(geo[:,5].mean())),
             sampled_tokens=tokens,outside_fsq_codes=outside,fid_by_prefix_sample_count=convergence)
 return result

def generation(condition,model,stats,motion_encoder,text_encoder,metrics):
 target=OUT/(condition+'.json')
 if target.exists():return json.loads(target.read_text())
 seed();model.cond_types=condition.split('_');loader=data_loader(condition)
 folder=OUT/condition;folder.mkdir(exist_ok=True)
 started=time.monotonic();computed=0;count=0
 with torch.no_grad():
  for i,data in enumerate(loader):
   if ARGS.max_batches and i>=ARGS.max_batches:break
   path=folder/f'{i:06d}.npz';sha=digest(data)
   if path.exists():
    with np.load(path) as saved:
     assert str(saved['input_sha256'])==sha,(condition,i,'dataset changed')
     torch.cuda.set_rng_state(torch.from_numpy(saved['cuda_rng'].copy()))
     count+=len(saved['pred'])
    continue
   lengths=data['motion_length'].cuda(non_blocking=True);gt=data['motion'].cuda(non_blocking=True)
   args=dict(motion=gt,motion_length=lengths,text_data=data['text_data'],mode='predict',return_ids=True)
   if condition=='text_traj':args['traj_data']=data['traj_data'].cuda(non_blocking=True)
   pred,_,ids=model(**args)
   if isinstance(pred,list):pred=torch.stack(pred)
   assert torch.isfinite(pred).all();valid=ids[ids>=0];assert (valid<model.pad_idx).all()
   n=len(pred);padding=torch.zeros(n,300-pred.shape[1],272,device='cuda')
   gp=motion_encoder(torch.cat((gt,padding),1),lengths).loc.cpu().numpy()
   pp=motion_encoder(torch.cat((pred,padding),1),lengths).loc.cpu().numpy()
   tt=text_encoder(data['text_data']).loc.cpu().numpy()
   # Use the exact NumPy recovery and released trajectory metric functions.
   p=metrics.recover_from_local_position((pred*stats['std_272']+stats['mean_272']).cpu().numpy(),22)
   g=metrics.recover_from_local_position((gt*stats['std_272']+stats['mean_272']).cpu().numpy(),22)
   error=np.linalg.norm(p[:,:,0]-g[:,:,0],axis=-1);lens=lengths.cpu().numpy();mask=np.arange(error.shape[1])[None]<lens[:,None]
   trajectory=metrics.calculate_trajectory_error(error,mask.astype(float),strict=True)
   trajectory=np.concatenate((trajectory,error[np.arange(n),lens-1,None]),axis=1)
   save_batch(path,input_sha256=sha,gt=gp,pred=pp,text=tt,trajectory=trajectory,lengths=lens,
              sampled_ids=ids.cpu().numpy(),captions=np.asarray(data['text_data']),
              tokens=int(valid.numel()),outside=int((valid>=1920).sum()),cuda_rng=torch.cuda.get_rng_state().cpu().numpy())
   count+=n;computed+=1;progress(condition,i,min(len(loader),ARGS.max_batches or len(loader)),count,started,computed)
 result=gen_metrics(folder,metrics,min(len(loader),ARGS.max_batches or len(loader)));result['status']='partial' if ARGS.max_batches else 'complete';result.update(seed=SEED,files=len(loader.dataset),wall_seconds=time.monotonic()-started)
 atomic_json(target,result);print('PAPER_CONDITION_COMPLETE',condition,json.dumps(result),flush=True)
 return result

def run(args):
 configure(args)
 torch.set_num_threads(8);seed()
 protocol=dict(seed=SEED,dtype='float32',sampling_steps=6,topk_filter_threshold=.9,
  text_files_per_batch=2,text_loader_workers=4,shuffle=True,reconstruction_files_per_batch=4,reconstruction_loader_workers=8,
  max_batches=args.max_batches,phases=args.phases,
  caption_crop='Released dataset random.choice/random.randint; mmengine.worker_init_fn rank0',
  retrieval='Released text-to-motion plus paper-described motion-to-text; final candidate group may have16 samples',
  caveat='Public single-GPU defaults. Historical paper launch command unavailable. No distributed-padding duplication.',
  split_hashes=spec['split_hashes'],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
  allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
 atomic_json(OUT/'protocol.json',protocol)
 model,stats,motion_encoder,text_encoder=load_runtime(args);metrics=official_metrics()
 results={}
 if 'reconstruction' in args.phases:results['reconstruction']=reconstruction(model,stats,metrics)
 for condition in ['text','text_traj']:
  if condition in args.phases:results[condition]=generation(condition,model,stats,motion_encoder,text_encoder,metrics)
 final=dict(status='partial' if args.max_batches else 'complete',protocol=protocol,**results)
 atomic_json(OUT/'results.json',final);atomic_json(OUT/'progress.json',dict(status=final['status']))
 from .report import write_report
 write_report(OUT,final)
 print('EVALUATION_FINISHED',final['status'],flush=True)
