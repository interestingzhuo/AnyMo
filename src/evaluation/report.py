"""Write a compact report using explicit fractions and units."""
from pathlib import Path
import json

def write_report(out,result):
 lines=['# AnyMo evaluation','',f"Status: {result['status']}",'',
        '| Condition | Samples | FID | R@1 text→motion (%) | MMDist | Diversity | ADE (cm) |',
        '|---|---:|---:|---:|---:|---:|---:|']
 for name in ['text','text_traj']:
  if name not in result:continue
  row=result[name]
  lines.append(f"| {name} | {row['sample_count']} | {row['fid']:.4f} | {row['R_text_to_motion'][0]*100:.4f} | {row['mmdist']:.4f} | {row['diversity']:.4f} | {row['trajectory']['ade_m']*100:.4f} |")
 if 'reconstruction' in result:
  row=result['reconstruction'];lines+=['',f"Tokenizer: {row['samples']} motions; root-relative MPJPE {row['mpjpe_mm']:.4f} mm (per-motion mean)."]
 lines+=['','R@2/3, both retrieval directions, real calibration, candidate counts, trajectory failure fractions and prefix FID are stored in results.json.',
         '', 'Pure-text ADE is distance to one reference path, which is not uniquely specified by text. No test-time fitting is applied here.',
         '', 'Use a fresh output directory after changing checkpoints, statistics, data, seed, source or sample limits.']
 (Path(out)/'REPORT.md').write_text('\n'.join(lines)+'\n')
