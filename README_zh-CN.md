# AnyMo

**Scaling Any-Modality Conditional Motion Generation with Masked Modeling**

[GitHub 代码](https://github.com/interestingzhuo/AnyMo) · [项目主页](https://interestingzhuo.github.io/anymo-page/) · [arXiv 论文](https://arxiv.org/abs/2605.29488) · [论文 PDF](https://arxiv.org/pdf/2605.29488) · [Hugging Face 模型](https://huggingface.co/human-motion-capture/AnyMo) · [OmniHuMo 数据集](https://huggingface.co/datasets/L-yiheng/OmniHuMo) · [处理后的测试数据](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo) · [English](README.md)

作者：Yiheng Li、Zhuo Li、Ruibing Hou、Yingjie Chen、Hong Chang、Hao Liu、Shiguang Shan。

AnyMo 结合残差有限标量量化（R-FSQ）动作 tokenizer 和掩码 Transformer，支持文本、语音、音乐、轨迹及其组合条件。配套 OmniHuMo 数据集包含约 320 万条动作序列、5,048 小时动作数据。

本发布候选首先整理了已经验证的 **Stage3 3B 文本／轨迹推理**。训练和评测代码一并保留，但完整训练、语音／音乐特征提取与完整评测还需独立验证，详见[发布状态](docs/RELEASE_STATUS.md)。

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

已有单条真实动作、3 个种子的验证记录见[验证说明](docs/VALIDATION.md)，不能将其当成完整数据集指标。数据约定见[数据格式](docs/DATA_FORMAT.md)，训练准备见[训练说明](docs/TRAINING.md)，论文引用见英文 README 和 `CITATION.cff`。

新整理入口的 CPU 回归和源文件检查已经通过，完整 GPU 整链路回归仍待完成：远端检查的自动权限审核连续超时。本包的验证说明已将历史复现数据与本次整理后的验证状态分别标明。

## 许可证

代码许可证由作者暂缓确定，见 [LICENSE_STATUS.md](LICENSE_STATUS.md)。论文、数据、第三方代码和模型权重的许可分别处理，不能相互替代。
