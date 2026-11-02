"""Download normalization and tokenizer files; checkpoint archives remain opt-in."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT)
    parser.add_argument('--stats-repo', default='human-motion-capture/processed_test_dataset_of_AnyMo')
    parser.add_argument('--stats-revision', default='8baf0d5da5f5dcc44a1281f2d251dbaa04118209')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    for name in ['mean_272.npy', 'std_272.npy', 'traj_mean.npy', 'traj_std.npy']:
        hf_hub_download(args.stats_repo, name, repo_type='dataset', revision=args.stats_revision,
                        local_dir=args.out / 'data')
    for name in ['config.json', 'spiece.model', 'tokenizer_config.json', 'special_tokens_map.json']:
        hf_hub_download('google/flan-t5-xl', name, local_dir=args.out / 'checkpoints/flan-t5-xl')
    print('Assets ready. Download the full AnyMo checkpoint separately; see README.md.')


if __name__ == '__main__':
    main()
