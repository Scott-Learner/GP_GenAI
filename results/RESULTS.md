# DFlash 最小推理对照实验

**结论：测得提速，但输出仅 15/30 组完全一致，不能宣称严格无损。**
全部 10 条提示词的总耗时比 4.1333× 含输出长度差异；额外取全部三轮输出一致的 5 条提示词（15 组，非最快样例筛选），双方均生成 2763 token，总耗时分别 94.912327s / 24.627344s，耗时比 **3.8539×**。

硬件：NVIDIA A100-PCIE-40GB，独占单卡 40GB，CPU affinity 12 核；实测驱动信息见 nvidia-smi-before.txt。
环境：Python 3.13.5，PyTorch 2.13.0+cu130（CUDA 13.0），Transformers 4.57.3；完整依赖见 dependencies.txt。
官方原版 DFlash commit：`7d2ea1e9828b058fbed637572f65ef86fb270195`（引入 DFlash 2 之前）。核心代码未修改。

目标 Qwen/Qwen3-8B 与 z-lab/Qwen3-8B-DFlash-b16；精度均为 BF16，attention 均为 SDPA，batch=1，单请求，无量化、无训练、无服务框架、无 torch.compile。
greedy（AR: do_sample=False、use_model_defaults=False；DFlash: temperature=0），thinking=False，block size=16，EOS=151645，每条最多 256 个新 token，总上下文上限 2048。实际提示词较短，输入及输出 token 数见 CSV。
普通自回归使用 Transformers generate(use_cache=True, logits_to_keep=1)。两模型始终共同驻留，避免加载次序影响。
10 条固定提示词覆盖代码、数学和一般问答；每种方式对全部提示词单独预热一次，正式每条重复 3 次，按提示词及轮次交替执行顺序（总计各 30 次）。
计时为输入 tensor 已准备好后调用生成到 GPU 完成的同步墙钟耗时，包含 prefill、draft、verify、decode 及生成调用内的分配与停止处理；不含分词、下载、模型加载、预热、CSV/JSON 序列化及打印。DFlash 官方 return_stats=True 的少量内部同步开销保留在计时内。
不报告官方纯 decode 时间：该旧版本会排除初次 drafter prefill，口径不等于完整 decode；所有结论均基于外层端到端时间。

|方式|正式生成 token 总数|正式总耗时 s|生成 tokens/s|峰值 allocated GiB|峰值 reserved GiB|采样设备占用峰值 GiB|
|---|---:|---:|---:|---:|---:|---:|
|普通自回归|5706|196.152431|29.090|17.261|17.342|18.333|
|DFlash|5610|47.456921|118.212|17.290|17.342|18.333|

总耗时比（AR / DFlash）：**4.1333×**；生成吞吐量比：4.0637×。吞吐量按总生成 token / 总耗时计算，输入 token 不计入。
每条提示词先合计其 3 次耗时再求加速比；10 条提示词的中位数：**4.7280×**，范围：1.5456–8.0304×。
token 序列完全一致的配对：**15/30**。每种方式自身三次重复一致性：{'ar': True, 'dflash': True}。
平均接受长度：5.6209 token/验证轮（采用官方 acceptance_lengths 定义，包含每轮确定的首 token；边界处可能含最终截断部分）。

显存说明：allocated 是 PyTorch 记录的精确张量分配高水位，reserved 是缓存分配器预留峰值；二者均含常驻目标及 drafter 权重。设备总占用来自 NVML 每 10ms 采样，含 CUDA 上下文、工作区等，因此不是连续时间的精确设备总峰值。缓存分配器不会在每次调用间清空，reserved 与采样占用可能包含此前调用保留的缓存。

|提示词|类别|AR 输出 token（各轮）|DFlash 输出 token（各轮）|AR 3轮合计 s|DFlash 3轮合计 s|加速比|逐 token 一致|
|---|---|---|---|---:|---:|---:|---|
|code_01|code|256/256/256|256/256/256|26.429523|3.948689|6.6932×|True|
|code_02|code|256/256/256|256/256/256|26.548877|6.254828|4.2445×|False|
|code_03|code|181/181/181|178/178/178|18.517583|4.162687|4.4485×|False|
|code_04|code|256/256/256|252/252/252|26.477284|4.360578|6.0720×|False|
|math_01|math|256/256/256|256/256/256|26.201776|4.770426|5.4925×|True|
|math_02|math|200/200/200|176/176/176|20.649770|2.571452|8.0304×|False|
|math_03|math|221/221/221|221/221/221|22.808433|4.554787|5.0076×|True|
|qa_01|qa|96/96/96|96/96/96|9.827046|5.112926|1.9220×|True|
|qa_02|qa|88/88/88|87/87/87|9.046590|5.480033|1.6508×|False|
|qa_03|qa|92/92/92|92/92/92|9.645549|6.240517|1.5456×|True|

模型修订：
- `Qwen/Qwen3-8B`: `b968826d9c46dd6066d109eabc6255188de91218`
- `z-lab/Qwen3-8B-DFlash-b16`: `9b41424b7109f9c5413454f481b09a82b85333f4`

权重通过同名 ModelScope 镜像下载，以锁定 Hugging Face 官方修订提供的 SHA256 逐文件验证；下载来源与校验值见项目 download_manifest.json。运行库从主机已安装分发只读复制到私有 venv，无共享 site-packages、无全局包修改，来源见 runtime_copy_provenance.json。
项目位于 /mnt/Playground/dflash-minimal。由于 playground 挂载仅 5GB、不支持符号链接，模型唯一缓存与独立环境位于 /tmp/dflash-minimal-runtime（本机 200GB overlay）；/tmp 可能随主机销毁或清理而消失，脚本和结果保存在 playground。

复跑（选择新的结果目录，不覆盖原始 CSV）：
```bash
/tmp/dflash-minimal-runtime/venv/bin/python /mnt/Playground/dflash-minimal/benchmark.py --output /mnt/Playground/dflash-minimal/results-rerun
/tmp/dflash-minimal-runtime/venv/bin/python /mnt/Playground/dflash-minimal/report.py /mnt/Playground/dflash-minimal/results-rerun
```

本结果只适用于当前硬件、软件后端、短提示词和本次 10 条小样本，不能外推到长上下文、批量请求或其他 GPU；不以论文加速倍数为目标。

存在输出差异，不能声称本次运行无损；首个差异及邻近 token 见 token_differences.json，进一步诊断见 discrepancy_diagnostics.json（若已生成）。不同输出长度还会影响耗时比的可比性。

一致性子集提示词：code_01, math_01, math_03, qa_01, qa_03。这是事后按一致性划分的补充结果，主结果仍覆盖全部 10 条固定提示词。

计时外差异诊断：五条提示词的两种输出均精确重现正式结果；普通解码在差异处确实使用单 token 查询和 KV cache，DFlash 使用 16-token 验证。停止规则和生成长度上限一致。

|提示词|首差异位置（从1计）|AR token 文本|DFlash token 文本|AR 第一/二候选 logit 间隔|DFlash 间隔|
|---|---:|---|---|---:|---:|
|code_02|143|`" closing"`|`" each"`|0.250000|0.000000|
|code_03|140|`" ("`|`".\n"`|0.000000|0.250000|
|code_04|101|`" "`|`"\n"`|0.250000|0.000000|
|math_02|89|`"}\n"`|`"}"`|0.125000|0.000000|
|qa_02|38|`" air"`|`" atmosphere"`|0.000000|0.250000|

这些首差异处的最高候选接近或并列，BF16 的数值差异可改变 greedy argmax。在相同 KV 张量的受控探针中，仅改变查询宽度（1 对 16）即在 math_02 重现 argmax 变化；即使强制相同 Math attention，此变化仍出现在 math_02。这支持查询形状及块式 KV 构造路径产生的 BF16 数值差异这一解释，尚未逐层隔离全部矩阵乘及 attention 算子的贡献。
同一 SDPA 配置会按形状选择不同内核：探针的 q=1 使用 aten::_scaled_dot_product_flash_attention，q=16 使用 aten::_scaled_dot_product_efficient_attention。二者均可在本 A100 运行。强制 Math 仅用于计时外诊断，未改变正式基准。
平均接受长度在代码/数学提示词约为 5.84–9.83，普通问答约为 2.09–2.69；较低接受长度与较低提速相伴。硬件、Python/Transformers 调度与具体 attention 内核的独立因果贡献未在此小实验中分离。
未为追求一致性或更高数字修改 DFlash 核心、改精度、改 block size 或重新选择正式样例。完整 logits 及探针操作名称见 discrepancy_diagnostics.json。
