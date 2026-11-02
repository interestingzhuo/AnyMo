"""Download-stream integrity and asset preflight regressions (CPU-only)."""
import importlib.util
import io
import hashlib
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
prepare=module('prepare_evaluation',ROOT/'tools/prepare_evaluation.py')
cli=module('evaluate_cli',ROOT/'tools/evaluate.py')

class EvaluationAssetsTest(unittest.TestCase):
 def test_shards_cross_boundaries_and_verify_all_bytes(self):
  with tempfile.TemporaryDirectory() as temp:
   paths=[];expected={};content=b'first-shard-content\x00second-shard'
   for index,data in enumerate([content[:19],b'',content[19:]]):
    path=Path(temp)/str(index);path.write_bytes(data);paths.append(path);expected[path.name]=hashlib.sha256(data).hexdigest()
   reader=prepare.ShardReader(paths,expected)
   with io.BufferedReader(reader,buffer_size=7) as buffered:self.assertEqual(buffered.read(),content);self.assertEqual(len(reader.verified),3)
 def test_corrupt_shard_rejected_at_eof(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'part';path.write_bytes(b'corrupt')
   with io.BufferedReader(prepare.ShardReader([path],{'part':hashlib.sha256(b'expected').hexdigest()})) as source:
    with self.assertRaisesRegex(ValueError,'checksum mismatch'):source.read()
 def test_split_traversal_rejected_before_loading_data(self):
  import argparse
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp);(path/'omnihumo_motion_caption_dynamic_test.txt').write_text('data/../outside.pkl\n')
   args=argparse.Namespace(checkpoint=path/'model',config=path/'config',evaluator_config=path/'ec',evaluator_checkpoint=path/'ep',tokenizer_checkpoint=path/'vq',text_backbone=path,assets_dir=path,split_dir=path,data_root=path,phases=['text'])
   with self.assertRaisesRegex(ValueError,'Unsafe test split'):cli.validate(args)
if __name__=='__main__':unittest.main()
