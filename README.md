# GP_GenAI

DFlash 最小推理对照实验：在单张 NVIDIA A100 40GB 上，对比 Qwen3-8B 普通自回归与原版 DFlash 推测解码。

## 实验结果

- 10 条固定提示词（代码、数学、问答），每种方法各重复 3 次，共 30 组配对。
- 全部样本总耗时比：**4.1333×**；吞吐量比：**4.0637×**。
- 输出逐 token 完全一致：**15/30 组**，因此不能宣称严格无损；全部样本的输出长度差异也影响耗时比的可比性。
- 全部三轮输出一致的 5 条提示词子集：双方各生成 2763 token，总耗时比 **3.8539×**。这是事后按一致性划分的补充结果。

完整口径、逐提示词结果和限制见 [实验报告](results/RESULTS.md)。这里只记录已有实验，没有为上传重新运行 GPU 基准。

## 文件

- `benchmark.py`、`generation_policy.py`、`metrics.py`：实验与指标代码。
- `report.py`、`diagnose_differences.py`：报告生成与输出差异诊断。
- `prompts.json`、`model_revisions.json`：固定提示词与模型版本。
- `results/`：原始 CSV、生成输出 JSONL、汇总、显存记录及诊断。
- `preliminary-invalid-warmup/`：已作废的早期预热记录，不用于正式结论。
- `official/`：官方 DFlash 子模块，固定到 `7d2ea1e9828b058fbed637572f65ef86fb270195`，保留上游 MIT 许可证。
- 下载清单、依赖、环境来源及日志保留在根目录，便于审计。

## 获取代码

```bash
git clone --recurse-submodules https://github.com/Scott-Learner/GP_GenAI.git
cd GP_GenAI
```

已经克隆的仓库执行 `git submodule update --init --recursive`。GitHub 的源码 ZIP 不包含子模块内容；从压缩包开始时，在项目根目录执行：

```bash
git clone https://github.com/z-lab/dflash.git official
git -C official checkout 7d2ea1e9828b058fbed637572f65ef86fb270195
```

## 环境和复跑

原实验使用 Python 3.13.5、PyTorch 2.13.0+cu130、Transformers 4.57.3，BF16、SDPA、batch=1、greedy、block size=16；完整依赖见 `results/dependencies.txt`。

`setup.sh` 和 `bootstrap_runtime.py` 是原实验主机的环境恢复脚本：要求主机已经安装对应 PyTorch，然后复制其依赖到 `/tmp/dflash-minimal-runtime/venv`；这不是任意主机通用的安装器。模型权重未上传，可用 `download_models.py` 按固定版本下载并校验，运行时位于 `/tmp/dflash-minimal-runtime/models`。

在满足原主机环境前提时：

```bash
bash setup.sh
bash run.sh ./results-rerun
```

请使用新的结果目录，避免覆盖正式数据。模型权重和缓存不属于此仓库。报告中的绝对路径描述原实验环境；计时和一致性结论仅适用于本次硬件、软件和小样本。

## 本地测试

```bash
/tmp/dflash-minimal-runtime/venv/bin/python -m unittest test_metrics.py test_generation_policy.py
```
