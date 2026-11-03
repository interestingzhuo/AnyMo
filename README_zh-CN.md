# AnyMo

**Scaling Any-Modality Conditional Motion Generation with Masked Modeling**

[项目主页](https://interestingzhuo.github.io/anymo-page/) · [arXiv 论文](https://arxiv.org/abs/2605.29488) · [Hugging Face 模型](https://huggingface.co/human-motion-capture/AnyMo) · [OmniHuMo 数据集](https://huggingface.co/datasets/L-yiheng/OmniHuMo) · [处理后的测试数据](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo) · [English](README.md)

作者：Yiheng Li、Zhuo Li、Ruibing Hou、Yingjie Chen、Hong Chang、Hao Liu、Shiguang Shan。

AnyMo 结合残差有限标量量化（R-FSQ）动作 tokenizer 和掩码 Transformer，支持文本、语音、音乐、轨迹及其组合条件。配套 OmniHuMo 数据集包含约 320 万条动作序列、5,048 小时动作数据。

本仓库提供 AnyMo 模型、训练与评测代码，以及已发布 Stage3 3B 检查点的文本／轨迹推理接口。

后续扩展计划见 [TODO](TODO.md)。

## 安装

使用 Linux、Python 3.11、CUDA 和 PyTorch 2.5.1。本推理入口暂不支持 CPU 或 Apple MPS。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python tools/download_assets.py
```

3B 完整检查点约 17.6 GB，包含冻结的 T5 和动作 tokenizer 权重；不需要重复下载 T5 权重。建议预留至少 24 GB 显存和足够加载完整检查点的主机内存。

模型与归一化数据仓库如需授权，请先获得仓库访问权限，再用 `hf auth login` 登录只读 Token；不要把 Token 写入代码。

```bash
hf download human-motion-capture/AnyMo \
  --revision 8b3f5719d4b0a0084c09a1c038266c511e6cb09f \
  --include 'work_dirs.tar.gz.part_*' --local-dir downloads/models
cat downloads/models/work_dirs.tar.gz.part_* | tar -xz
```

完整压缩包约 57.3 GB，请为解压预留额外空间。默认使用：

```text
work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth
configs/inference/anymo_3b.py
```

该配置对应 24 层、32 个注意力头、3200 隐藏维度。其他模型需使用匹配配置；不能仅修改文件名切换模型。训练配置中的 7B 定义不代表已发布模型包中含有 7B 检查点。

## 文本生成动作

```bash
python tools/infer.py \
  --checkpoint work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth \
  --prompt "A person steadily walks forward, alternately stepping with the left and right foot while swinging their arms." \
  --frames 300 --seed 3407 --out outputs/walking
```

动作帧率为 **30 fps**，长度为 **60～300 帧且须为 4 的倍数**。建议描述清晰的肢体动作和先后关系。原生采样默认 6 步；`--topk-threshold 0.9` 表示保留分数最高的 10%，不是 top-p。

## 文本＋轨迹

轨迹为未归一化的骨盆世界坐标 `(T, 3)`，单位米，坐标顺序 XYZ、Y 轴向上、30 fps。支持 `.npy`，或包含 `target_traj` 字段的 `.npz`。程序会应用发布的轨迹均值和标准差，不会自动重采样、旋转、缩放或对齐坐标。

```bash
python tools/infer.py \
  --checkpoint work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth \
  --prompt "A person turns their body while swinging both arms." \
  --trajectory path/to/trajectory.npy \
  --tta-steps 400 --tta-lr 0.0002 --out outputs/trajectory
```

省略 `--prompt` 可运行纯轨迹条件；省略 `--tta-steps` 则只做原生条件生成。`tools/test_traj.py` 是要求轨迹输入、默认优化 400 步的兼容入口。

轨迹优化固定模型权重，只更新解码前的连续 latent，最小化骨盆轨迹 XYZ 均方误差。默认 AdamW 参数为 `lr=2e-4`、`betas=(0.5, 0.9)`、`weight_decay=1e-6`。轨迹拟合没有加入脚底接触约束，跟随更准确并不等于消除滑步；增大学习率可能发散。

## 输出

- `native.npz`：原生动作，包含 `normalized`、`motion`、`joints`、`tokens`、`prompt`、`fps`；有轨迹时还包含 `target_traj`。
- `optimized.npz`：成功优化后的动作；其中的 tokens 是初始化时采样的码，没有把优化后的 latent 重新量化。
- `report.json`：参数、随机种子、加载状态及轨迹误差。
- `resolved_config.py`：本次实际使用的配置。

## Train

数据下载、目录布局、记录格式、归一化与划分配置见 [Data preparation](docs/DATA_PREPARATION.md)。

按[训练说明](docs/TRAINING.md)准备完整训练数据、归一化统计和预训练编码器，选择与初始化检查点匹配的配置。

```bash
python -m pip install -r requirements/evaluation.txt
bash tools/dist_train.sh configs/tokenizer_r_fsq.py 8 --work-dir work_dirs/tokenizer_r_fsq_2048cd_4res_200e
bash tools/dist_train.sh configs/train/anymo_stage_1_3b.py 8 --work-dir work_dirs/stage1_3b
bash tools/dist_train.sh configs/train/anymo_stage_2_3b.py 8 --work-dir work_dirs/stage2_3b
bash tools/dist_train.sh configs/train/anymo_stage_3_3b.py 8 --work-dir work_dirs/stage3_3b
```

## Evaluation

按[评测说明](docs/EVALUATION.md)准备检查点、测试数据和评测编码器。

```bash
# 文本＋轨迹（eval_text.py 的默认条件）
python tools/test.py configs/eval_text.py \
  work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth

# Tokenizer 重建
python tools/test.py configs/tokenizer_r_fsq.py \
  work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth
```

数据约定见[数据格式](docs/DATA_FORMAT.md)。

## 许可证

作者已选定 [Apache-2.0](LICENSE) 用于 AnyMo 自有代码。**第三方组件保留原有条款，其中 SnapMoGen 派生代码仍受非商业研究限制**，不能将整个仓库理解为仅受 Apache-2.0 约束。详见[许可范围](LICENSE_STATUS.md)、[NOTICE](NOTICE)与[第三方说明](THIRD_PARTY_NOTICES.md)。模型权重、数据和媒体许可独立。
