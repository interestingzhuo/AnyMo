# AnyMo

**Scaling Any-Modality Conditional Motion Generation with Masked Modeling**

[项目主页](https://interestingzhuo.github.io/anymo-page/) · [arXiv 论文](https://arxiv.org/abs/2605.29488) · [Hugging Face 模型](https://huggingface.co/human-motion-capture/AnyMo) · [OmniHuMo 数据集](https://huggingface.co/datasets/L-yiheng/OmniHuMo) · [处理后的测试数据](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo) · [English](README.md)

作者：Yiheng Li、Zhuo Li、Ruibing Hou、Yingjie Chen、Hong Chang、Hao Liu、Shiguang Shan。

AnyMo 结合残差有限标量量化（R-FSQ）动作 tokenizer 和掩码 Transformer，支持文本、语音、音乐、轨迹及其组合条件。配套 OmniHuMo 数据集包含约 320 万条动作序列、5,048 小时动作数据。

本发布候选首先整理了已经验证的 **Stage3 3B 文本／轨迹推理**。完整文本／轨迹生成及 tokenizer 重建评测已完成；完整训练、语音／音乐特征提取及其评测尚未重新验证，详见[发布状态](docs/RELEASE_STATUS.md)。

## 安装

推荐 Linux、Python 3.11、CUDA 和 PyTorch 2.5.1。已在 H20 上运行 CUDA 12.4 版本；原依赖快照使用 CUDA 12.1。本推理入口暂不支持 CPU 或 Apple MPS。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python tools/download_assets.py
```

3B 完整检查点约 17.6 GB，包含冻结的 T5 和动作 tokenizer 权重；不需要重复下载 T5 权重。H20 上显存分配约 17 GiB，建议预留 24 GB 以上显存和充足主机内存。64 GB 主机内存是加载完整检查点的实用起点，并非所有平台的最低配置。

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

该配置对应 24 层、32 个注意力头、3200 隐藏维度。其他模型需使用匹配配置；不能仅修改文件名切换模型。训练配置中的 7B 定义不代表已验证模型包中含有 7B 检查点。

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

轨迹优化固定模型权重，只更新解码前的连续 latent，最小化骨盆轨迹 XYZ 均方误差。默认 AdamW 参数为 `lr=2e-4`、`betas=(0.5, 0.9)`、`weight_decay=1e-6`。本候选修复了原轨迹恢复函数的反向传播问题。轨迹拟合没有加入脚底接触约束，跟随更准确并不等于消除滑步；增大学习率可能发散。

## 输出和验证

- `native.npz`：原生动作，包含 `normalized`、`motion`、`joints`、`tokens`、`prompt`、`fps`；有轨迹时还包含 `target_traj`。
- `optimized.npz`：成功优化后的动作；其中的 tokens 是初始化时采样的码，没有把优化后的 latent 重新量化。
- `report.json`：参数、随机种子、加载状态及轨迹误差。
- `resolved_config.py`：本次实际使用的配置。

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

完整默认评测已完成：每种生成条件 **51,408 条动作**，重建 **182,640 条动作**。

| 指标 | 论文归档值 | 本次全量 |
|---|---:|---:|
| 纯文本 FID ↓ | 55.5974 | **55.6296** |
| 纯文本 R@1 ↑ | 75.0019% | **75.1167%** |
| 文本＋轨迹 FID ↓ | 41.4394 | **41.1119** |
| 文本＋轨迹 R@1 ↑ | 76.9881% | **77.0386%** |
| 文本＋轨迹 ADE ↓ | 27.1594 cm | **27.3120 cm** |
| tokenizer MPJPE ↓ | 13.2 mm | **13.2649 mm** |

使用单卡 H20、FP32、种子3407、原生6步采样；评测未加入轨迹优化。R@1 沿用官方代码的文本→动作方向，两个方向均保存在原始结果中。历史测试启动命令无法找回，本次按作者确认的公开默认配置运行，最后一批有16个检索候选。

```bash
python -m pip install -r requirements/benchmark.txt
python tools/prepare_evaluation.py
python tools/evaluate.py --check-only
python tools/evaluate.py --out outputs/full_evaluation
```

[评测说明](docs/EVALUATION.md)包含数据准备、路径覆盖、冒烟检查和断点续跑命令；[全量报告](validation/full_20261102/REPORT.md)与[原始指标](validation/full_20261102/results.json)已随仓库保存。全量结果验证了核心模型与评测逻辑，参数化新入口的检查状态见[验证说明](docs/VALIDATION.md)。数据约定见[数据格式](docs/DATA_FORMAT.md)。

## 许可证

作者已选定 [Apache-2.0](LICENSE) 用于 AnyMo 自有代码。**第三方组件保留原有条款，其中 SnapMoGen 派生代码仍受非商业研究限制**，不能将整个仓库理解为仅受 Apache-2.0 约束。详见[许可范围](LICENSE_STATUS.md)、[NOTICE](NOTICE)与[第三方说明](THIRD_PARTY_NOTICES.md)。模型权重、数据和媒体许可独立。
