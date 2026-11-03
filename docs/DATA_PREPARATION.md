# Data preparation

Run the commands below from the repository root after installing the dependencies in [Train](TRAINING.md). Training uses processed motion records and explicit split lists. Data access and dataset licenses are separate from the code license.

## 1. Select the data release

| Source | Contents / purpose |
|---|---|
| [OmniHuMo](https://huggingface.co/datasets/L-yiheng/OmniHuMo) | Full dataset release: motion, audio, precomputed audio features, splits and processing code |
| [Processed AnyMo test data](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo) | Prepared evaluation records and normalization arrays; not the full training corpus |
| MotionMillion | Additional data referenced by the tokenizer template; obtain separately or configure the tokenizer to use your available training/validation data |

If a repository requires access, accept its terms and obtain access before running `hf auth login`. Download only the assets needed for your task.

### Full dataset and processing assets

The OmniHuMo release lists separate motion, audio-feature and waveform archives. For motion and precomputed features:

```bash
hf download L-yiheng/OmniHuMo --repo-type dataset \
  --local-dir downloads/omnihumo \
  --include 'omnihumo_v0.tar.gz.part_*' 'audio_feat.tar.gz.part_*' \
  'split.tar.gz' 'process_code/**'
```

Raw waveform archives (`audio.tar.gz.part_*`) are optional when using precomputed features. The official processing assets include `process_code/process_motion_272.py` and `process_code/utils/audio_extractor.py`. Use that dataset release's processing instructions and required pretrained/body-model assets when converting raw data. The AnyMo training code consumes the final records described below; it does not automatically convert raw body parameters, match captions/audio to motions or extract waveform features.

Inspect an archive's top-level paths before choosing the extraction destination:

```bash
cat downloads/omnihumo/omnihumo_v0.tar.gz.part_* \
  | tar -tzf - > downloads/omnihumo/motion_contents.txt
cat downloads/omnihumo/audio_feat.tar.gz.part_* \
  | tar -tzf - > downloads/omnihumo/audio_feature_contents.txt
tar -tzf downloads/omnihumo/split.tar.gz
```

Concatenate all parts of each archive in filename order, then extract to separate staging directories:

```bash
mkdir -p downloads/omnihumo/unpacked/motion \
  downloads/omnihumo/unpacked/audio_features downloads/omnihumo/unpacked/splits
cat downloads/omnihumo/omnihumo_v0.tar.gz.part_* \
  | tar -xzf - -C downloads/omnihumo/unpacked/motion
cat downloads/omnihumo/audio_feat.tar.gz.part_* \
  | tar -xzf - -C downloads/omnihumo/unpacked/audio_features
tar -xzf downloads/omnihumo/split.tar.gz -C downloads/omnihumo/unpacked/splits
```

Arrange the processed records under the layout below. The training lists require final record files, so extracting an archive alone is not proof that its contents already match the training schema. Keep the released train/test membership when converting and arranging data.

### Prepared evaluation records

The prepared test archive contains an `omnihumo/` directory, so extract it under `data/`:

```bash
hf download human-motion-capture/processed_test_dataset_of_AnyMo \
  --repo-type dataset --revision 8baf0d5da5f5dcc44a1281f2d251dbaa04118209 \
  --include 'omnihumo.tar.gz.part_*' --local-dir downloads/test_data
mkdir -p data
cat downloads/test_data/omnihumo.tar.gz.part_* | tar -xzf - -C data
```

Do not add these evaluation records to training lists. Prepared test data supplies only test members of the directories below.

## 2. Arrange files and normalization

The supplied lists use this processed layout:

```text
data/
  mean_272.npy                  # (272,)
  std_272.npy                   # (272,)
  traj_mean.npy                # (3,)
  traj_std.npy                 # (3,)
  omnihumo/
    motion_pkl/
    motion_caption_dynamic_pkl/
    motion_caption_static_pkl/
    motion_caption_speak_pkl/
    motion_caption_dance_pkl/
  motionmillion/               # Only when using the additional tokenizer data
    motion_272_batch_16/
    mini_test.pkl
configs/data_paths/
  ...                          # One processed pickle path per line
```

Download the published normalization arrays with the existing asset tool:

```bash
python tools/download_assets.py
```

This also downloads FLAN-T5 configuration/tokenizer files. It does not download the dataset, model checkpoints or full Stage1 pretrained T5 weights. Use the statistics associated with the released checkpoint for inference/evaluation. Store unnormalized motion/features in the records: the loaders apply motion and trajectory normalization themselves. See [data conventions](DATA_FORMAT.md).

## 3. Match the record schema

Each training pickle is loaded with `joblib.load` and contains a **list of dictionaries**, not a single dictionary. Use trusted pickle files only.

| Field | Type / shape | Required by |
|---|---|---|
| `motion_272` | Finite floating-point array `(T, 272)`, 30 fps, unnormalized | All training stages |
| `captions` | Nonempty list of strings | Stage1, Stage2 and Stage3 |
| `audio_feat` | Finite floating-point array `(A, 512)`, 75 Hz, synchronized with the motion | Every Stage2 record; audio-conditioned Stage3 records |

For example, a two-second audio-conditioned record has 60 motion frames and approximately 150 audio-feature frames:

```python
import joblib
import numpy as np

# motion_array, caption_strings and aligned_audio_features come from your
# processed dataset. Preserve their coordinate system and common time origin.
records = [{
    "motion_272": np.asarray(motion_array, dtype=np.float32),
    "captions": list(caption_strings),
    "audio_feat": np.asarray(aligned_audio_features, dtype=np.float32),
}]
joblib.dump(records, "data/custom/train/0000001.pkl")
```

Omit `audio_feat` for motion/text-only records. Stage2's `AnyMoDataset` always reads audio and captions; Stage3's `AnyMoDataset_v2` supports records without audio but still reads captions. Trajectories are recovered from `motion_272` by the loaders; a separate `root_traj` field is not required.

Prepare motion records with at least 60 frames. With the standard configuration, the loader rounds the selected motion length down to a multiple of four, crops at most 300 frames and pads to the configured maximum. Longer source records are allowed; they need aligned audio covering the selected crop. The source record itself need not have a frame count divisible by four. The tokenizer's separate `MotionDataset` validation loader filters for lengths strictly greater than `min_motion_length`.

Audio is cropped at the same time origin using the 75/30 fps ratio. Its selected length is rounded down to a multiple of eight feature frames: `floor((motion_length * 75 / 30) / 8) * 8`. Preserve the published feature representation; a waveform or an arbitrary 512-dimensional embedding cannot substitute for `audio_feat`.

## 4. Configure split lists

Each nonempty line is a pickle path relative to the **repository root**, not relative to the list file:

```text
data/omnihumo/motion_caption_dynamic_pkl/0010114.pkl
data/omnihumo/motion_caption_dynamic_pkl/0010115.pkl
```

Use the supplied `configs/data_paths/` lists when your processed files match the release. For custom data, first separate train/validation/test records according to your dataset protocol. Then create a list for one already separated directory:

```python
from pathlib import Path

paths = sorted(Path("data/custom/train").rglob("*.pkl"))
assert paths, "No processed training files found"
with Path("configs/data_paths/my_text_train.txt").open("x") as output:
    output.write("\n".join(path.as_posix() for path in paths) + "\n")
```

Do not recursively scan a directory containing both train and test records into one training list. Keep sequences from the same source clip/person grouped according to the intended split protocol. Avoid empty lines, duplicate paths and empty record lists. Batch size counts pickle files; the collator expands their records into the actual motion batch.

| Configuration | Dataset / list fields |
|---|---|
| Tokenizer | `AnyMoMotionDataset`: `train_dataloader.dataset.data_path_file` (list of list-file paths) |
| Stage1 | `AnyMoTextDataset`: `train_dataloader.dataset.data_path_file` (list of list-file paths) |
| Stage2 | `AnyMoDataset`: `train_dataloader.dataset.data_path_file` (speech/dance list-file paths) |
| Stage3 | `AnyMoDataset_v2`: `audio_path_file` (list of list-file paths), `caption_path_file` (one list-file path) |

**Stage1 currently inherits an empty `data_path_file` list. Set it before training**, for example:

```bash
python tools/train.py configs/train/anymo_stage_1_3b.py \
  --work-dir work_dirs/stage1_3b \
  --cfg-options \
  train_dataloader.dataset.data_path_file="['configs/data_paths/my_text_train.txt']"
```

The tokenizer template also lists `configs/data_paths/mm_train.txt` and uses `data/motionmillion/mini_test.pkl` for validation. Supply those additional records, or replace the training list and configure the validation loader/evaluator consistently for your available dataset. Stage3's `caption_sample_ratio` controls the number of text-file entries via `int(number_of_caption_files * ratio)`; make sure this is nonzero when including a small custom text dataset.

## 5. Check the local paths and one record

After preparing a text training list, inspect it before launching the runner:

```python
from pathlib import Path
import joblib
import numpy as np

split = Path("configs/data_paths/my_text_train.txt")
lines = split.read_text().splitlines()
assert lines and all(line.strip() for line in lines), "Empty split or blank lines"
paths = [Path(line.strip()) for line in lines]
assert len({path.resolve() for path in paths}) == len(paths), "Duplicate paths"
missing = [str(path) for path in paths if not path.is_file()]
assert not missing, f"Missing {len(missing)} files; first paths: {missing[:5]}"

records = joblib.load(paths[0])
assert isinstance(records, list) and records
for record in records:
    motion = np.asarray(record["motion_272"])
    assert motion.ndim == 2 and motion.shape[1] == 272 and len(motion) >= 60
    assert np.isfinite(motion).all()
    assert isinstance(record["captions"], list) and record["captions"]
    assert all(isinstance(text, str) for text in record["captions"])
    if "audio_feat" in record:
        audio = np.asarray(record["audio_feat"])
        assert audio.ndim == 2 and audio.shape[1] == 512 and np.isfinite(audio).all()
        assert abs(len(audio) - len(motion) * 75 / 30) <= 8, "Check audio/motion alignment"
print(f"{len(paths)} files found; first file contains {len(records)} compatible records")
```

This inspects the first file's records; repeat the schema check over the complete split for custom datasets. Audio-stage lists must contain `audio_feat` in every audio record. Metadata checks do not establish actual audio/motion synchronization, so verify source pairing and timestamps during preprocessing.

Then follow [Train](TRAINING.md) for checkpoint initialization and launch commands, or [Evaluation](EVALUATION.md) for evaluator assets and test commands.
