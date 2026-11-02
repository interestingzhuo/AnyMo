"""Download pinned test shards and safely extract the published evaluation lists."""
import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA_REPO='human-motion-capture/processed_test_dataset_of_AnyMo'
DATA_REVISION='8baf0d5da5f5dcc44a1281f2d251dbaa04118209'
BACKBONE_REVISION='12040accade4e8a0f71eabdb258fecc2e7e948be'

class ShardReader(io.RawIOBase):
    def __init__(self,parts,expected):
        super().__init__();self.parts=iter(parts);self.expected=expected;self.current=None;self.name=None;self.digest=None;self.verified=[]
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                path=next(self.parts,None)
                if path is None:return 0
                self.name=path.name;self.current=path.open('rb');self.digest=hashlib.sha256()
            count=self.current.readinto(buffer)
            if count:self.digest.update(memoryview(buffer)[:count]);return count
            digest=self.digest.hexdigest();self.current.close();self.current=None
            if digest!=self.expected[self.name]:raise ValueError('Shard checksum mismatch: '+self.name)
            self.verified.append(dict(name=self.name,sha256=digest));print('Verified',self.name,flush=True)
    def close(self):
        if self.current is not None:self.current.close()
        super().close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path,default=ROOT)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data')
    parser.add_argument('--archives-dir',type=Path,default=ROOT/'downloads/test-data')
    parser.add_argument('--split-dir',type=Path,default=ROOT/'configs/data_paths')
    parser.add_argument('--backbone-only',action='store_true')
    args=parser.parse_args()
    from huggingface_hub import HfApi,hf_hub_download,snapshot_download
    backbone='distilbert/distilbert-base-uncased'
    # Configuration/tokenizer only: learned DistilBERT parameters are in the TEMOS checkpoint.
    revision=BACKBONE_REVISION or HfApi().model_info(backbone).sha
    snapshot_download(backbone,revision=revision,allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'],local_dir=args.assets_dir/'checkpoints/distilbert-base-uncased')
    if args.backbone_only:return
    names=['omnihumo_vq_test','omnihumo_motion_caption_dynamic_test','omnihumo_motion_caption_static_test','dance_test','speak_test']
    selected=set();split_hashes={}
    for name in names:
        path=args.split_dir/(name+'.txt');split_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            value=line.strip().removeprefix('data/')
            if not value:continue
            if Path(value).is_absolute() or '..' in Path(value).parts:raise ValueError('Unsafe test split: '+value)
            selected.add(value)
    info=HfApi().dataset_info(DATA_REPO,revision=DATA_REVISION,files_metadata=True)
    shards=sorted([x for x in info.siblings if x.rfilename.startswith('omnihumo.tar.gz.part_')],key=lambda x:x.rfilename)
    if len(shards)!=7:raise ValueError('Expected seven pinned dataset shards')
    parts=[Path(hf_hub_download(DATA_REPO,x.rfilename,repo_type='dataset',revision=DATA_REVISION,local_dir=args.archives_dir)) for x in shards]
    reader=ShardReader(parts,{x.rfilename:x.lfs.sha256 for x in shards});buffered=io.BufferedReader(reader,buffer_size=4*1024**2)
    import gzip
    found=[]
    # Explicit regular-file extraction; archive symlinks and unselected members are never extracted.
    with gzip.GzipFile(fileobj=buffered,mode='rb') as decoded,tarfile.open(fileobj=decoded,mode='r|',bufsize=4*1024**2) as archive:
        for member in archive:
            name=member.name.removeprefix('./')
            if name not in selected or not member.isfile():continue
            target=args.data_root/name;target.parent.mkdir(parents=True,exist_ok=True)
            # Reject pre-existing symlink destinations as well as archive traversal.
            if not target.resolve().is_relative_to(args.data_root.resolve()):raise ValueError('Destination leaves data root')
            temporary=target.with_suffix('.partial')
            if temporary.is_symlink():raise ValueError('Temporary extraction path is a symlink')
            with archive.extractfile(member) as source,temporary.open('wb') as output:shutil.copyfileobj(source,output,length=4*1024**2)
            if temporary.stat().st_size!=member.size:raise ValueError('Truncated file: '+name)
            temporary.replace(target);found.append(name)
            if len(found)%500==0:print('Extracted',len(found),'/',len(selected),flush=True)
    while buffered.read(4*1024**2):pass
    missing=sorted(selected-set(found));report=dict(dataset_revision=DATA_REVISION,backbone_revision=revision,split_hashes=split_hashes,files=len(found),missing=missing,verified_shards=reader.verified)
    args.data_root.mkdir(parents=True,exist_ok=True);(args.data_root/'evaluation_manifest.json').write_text(json.dumps(report,indent=2)+'\n');buffered.close()
    if missing:raise ValueError(f'Missing {len(missing)} test files')
    print('All test files extracted and all source shards verified.',flush=True)
if __name__=='__main__':main()
