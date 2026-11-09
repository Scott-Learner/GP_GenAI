"""Audit raw measurements and regenerate the short Chinese result report."""
import argparse,csv,hashlib,json,pathlib,statistics
from metrics import compare_tokens,summarize
PROJECT=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('results',nargs='?',type=pathlib.Path,default=PROJECT/'results');args=p.parse_args();out=args.results.resolve()
records=[json.loads(s) for s in (out/'outputs.jsonl').read_text().splitlines()]
with (out/'raw_measurements.csv').open() as f:csv_rows=list(csv.DictReader(f))
assert len(csv_rows)==len(records)==60
for c,r in zip(csv_rows,records):
    assert c['token_sha256']==hashlib.sha256(json.dumps(r['tokens']).encode()).hexdigest()
    assert int(c['generated_tokens'])==r['generated_tokens']==len(r['tokens'])
    assert float(c['elapsed_s'])==r['elapsed_s']>0
    assert abs(float(c['tokens_per_s'])-r['generated_tokens']/r['elapsed_s'])<1e-8
comparisons=[]
for prompt in sorted({r['prompt_id'] for r in records}):
    for repeat in range(3):
        pair={r['method']:r for r in records if r['prompt_id']==prompt and r['repeat']==repeat}
        comparison=compare_tokens(pair['ar']['tokens'],pair['dflash']['tokens'])
        assert comparison['equal']==pair['ar']['equal_to_pair']==pair['dflash']['equal_to_pair']
        if not comparison['equal']:
            pos=comparison['first_difference']
            comparisons.append({'prompt_id':prompt,'repeat':repeat,'first_difference':pos,
                                'ar_length':len(pair['ar']['tokens']),'dflash_length':len(pair['dflash']['tokens']),
                                'ar_nearby':pair['ar']['tokens'][max(0,pos-5):pos+6],
                                'dflash_nearby':pair['dflash']['tokens'][max(0,pos-5):pos+6]})
s=summarize(records);saved=json.loads((out/'summary.json').read_text())
for k,v in s.items():assert saved[k]==v,(k,saved[k],v)
manifest=json.loads((out/'run_manifest.json').read_text());assert manifest['status']=='complete'
assert not manifest['official_worktree_dirty']
assert s['equal_pairs']+len(comparisons)==30
(out/'token_differences.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
methods=s['methods'];ar=methods['ar'];df=methods['dflash'];lo,hi=s['prompt_speedup_range']
def gib(n):return n/1024**3
lines=[
'# DFlash 最小推理对照实验',
'',
f"硬件：{manifest['gpu']}，独占单卡 40GB，CPU affinity {len(manifest['cpu_affinity'])} 核；实测驱动信息见 nvidia-smi-before.txt。",
f"环境：Python {manifest['python']}，PyTorch {manifest['versions']['torch']}（CUDA {manifest['torch_cuda']}），Transformers {manifest['versions']['transformers']}；完整依赖见 dependencies.txt。",
f"官方原版 DFlash commit：`{manifest['official_commit']}`（引入 DFlash 2 之前）。核心代码未修改。",
'',
'目标 Qwen/Qwen3-8B 与 z-lab/Qwen3-8B-DFlash-b16；精度均为 BF16，attention 均为 SDPA，batch=1，单请求，无量化、无训练、无服务框架、无 torch.compile。',
'greedy（AR: do_sample=False、use_model_defaults=False；DFlash: temperature=0），thinking=False，block size=16，EOS=151645，每条最多 256 个新 token，总上下文上限 2048。实际提示词较短，输入及输出 token 数见 CSV。',
'普通自回归使用 Transformers generate(use_cache=True, logits_to_keep=1)。两模型始终共同驻留，避免加载次序影响。',
'10 条固定提示词覆盖代码、数学和一般问答；每种方式对全部提示词单独预热一次，正式每条重复 3 次，按提示词及轮次交替执行顺序（总计各 30 次）。',
'计时为输入 tensor 已准备好后调用生成到 GPU 完成的同步墙钟耗时，包含 prefill、draft、verify、decode 及生成调用内的分配与停止处理；不含分词、下载、模型加载、预热、CSV/JSON 序列化及打印。DFlash 官方 return_stats=True 的少量内部同步开销保留在计时内。',
'不报告官方纯 decode 时间：该旧版本会排除初次 drafter prefill，口径不等于完整 decode；所有结论均基于外层端到端时间。',
'',
'|方式|正式生成 token 总数|正式总耗时 s|生成 tokens/s|峰值 allocated GiB|峰值 reserved GiB|采样设备占用峰值 GiB|',
'|---|---:|---:|---:|---:|---:|---:|',
]
for label,m in [('普通自回归',ar),('DFlash',df)]:
    lines.append(f"|{label}|{m['generated_tokens']}|{m['total_elapsed_s']:.6f}|{m['tokens_per_s']:.3f}|{gib(m['peak_allocated_bytes']):.3f}|{gib(m['peak_reserved_bytes']):.3f}|{gib(m['peak_device_bytes']):.3f}|")
lines += ['',f"总耗时比（AR / DFlash）：**{s['total_time_speedup']:.4f}×**；生成吞吐量比：{s['throughput_ratio']:.4f}×。吞吐量按总生成 token / 总耗时计算，输入 token 不计入。",f"每条提示词先合计其 3 次耗时再求加速比；10 条提示词的中位数：**{s['prompt_speedup_median']:.4f}×**，范围：{lo:.4f}–{hi:.4f}×。",f"token 序列完全一致的配对：**{s['equal_pairs']}/30**。每种方式自身三次重复一致性：{saved['within_method_repeat_consistency']}。",f"平均接受长度：{s['mean_acceptance_length']:.4f} token/验证轮（采用官方 acceptance_lengths 定义，包含每轮确定的首 token；边界处可能含最终截断部分）。",'',
'显存说明：allocated 是 PyTorch 记录的精确张量分配高水位，reserved 是缓存分配器预留峰值；二者均含常驻目标及 drafter 权重。设备总占用来自 NVML 每 10ms 采样，含 CUDA 上下文、工作区等，因此不是连续时间的精确设备总峰值。缓存分配器不会在每次调用间清空，reserved 与采样占用可能包含此前调用保留的缓存。','',
'|提示词|类别|AR 输出 token（各轮）|DFlash 输出 token（各轮）|AR 3轮合计 s|DFlash 3轮合计 s|加速比|逐 token 一致|','|---|---|---|---|---:|---:|---:|---|']
for prompt in sorted(s['per_prompt_speedup']):
    rr={m:sorted([r for r in records if r['prompt_id']==prompt and r['method']==m],key=lambda r:r['repeat']) for m in ['ar','dflash']}
    token=lambda m:'/'.join(str(r['generated_tokens']) for r in rr[m])
    tm=lambda m:sum(r['elapsed_s'] for r in rr[m])
    lines.append(f"|{prompt}|{rr['ar'][0]['category']}|{token('ar')}|{token('dflash')}|{tm('ar'):.6f}|{tm('dflash'):.6f}|{s['per_prompt_speedup'][prompt]:.4f}×|{all(r['equal_to_pair'] for r in rr['ar'])}|")
lines+=['','模型修订：']
for model,meta in manifest['model_revisions'].items():lines.append(f"- `{model}`: `{meta['revision']}`")
lines+=['','权重通过同名 ModelScope 镜像下载，以锁定 Hugging Face 官方修订提供的 SHA256 逐文件验证；下载来源与校验值见项目 download_manifest.json。运行库从主机已安装分发只读复制到私有 venv，无共享 site-packages、无全局包修改，来源见 runtime_copy_provenance.json。',
'项目位于 /mnt/Playground/dflash-minimal。由于 playground 挂载仅 5GB、不支持符号链接，模型唯一缓存与独立环境位于 /tmp/dflash-minimal-runtime（本机 200GB overlay）；/tmp 可能随主机销毁或清理而消失，脚本和结果保存在 playground。',
'', '复跑（选择新的结果目录，不覆盖原始 CSV）：','```bash',f"/tmp/dflash-minimal-runtime/venv/bin/python {PROJECT}/benchmark.py --output {PROJECT}/results-rerun",f"/tmp/dflash-minimal-runtime/venv/bin/python {PROJECT}/report.py {PROJECT}/results-rerun",'```','',
'本结果只适用于当前硬件、软件后端、短提示词和本次 10 条小样本，不能外推到长上下文、批量请求或其他 GPU；不以论文加速倍数为目标。']
if comparisons:lines+=['','存在输出差异，不能声称本次运行无损；首个差异及邻近 token 见 token_differences.json，进一步诊断见 discrepancy_diagnostics.json（若已生成）。不同输出长度还会影响耗时比的可比性。']
else:lines+=['','全部正式配对在本次样本上 token 一致；这不是对所有输入或所有数值后端的无损性证明。']
# Supplement the complete ten-prompt report with the exact-output subset.
matched_ids=[pid for pid in sorted(s['per_prompt_speedup']) if all(r['equal_to_pair'] for r in records if r['prompt_id']==pid)]
additional={'matched_prompt_ids':matched_ids}
if matched_ids:
    subset=summarize([r for r in records if r['prompt_id'] in matched_ids],expected_prompts=len(matched_ids),repeats=3)
    additional['exact_output_subset']=subset
    lines[2:2]=[f"**结论：测得提速，但输出仅 {s['equal_pairs']}/30 组完全一致，不能宣称严格无损。**",f"全部 10 条提示词的总耗时比 {s['total_time_speedup']:.4f}× 含输出长度差异；额外取全部三轮输出一致的 {len(matched_ids)} 条提示词（{len(matched_ids)*3} 组，非最快样例筛选），双方均生成 {subset['methods']['ar']['generated_tokens']} token，总耗时分别 {subset['methods']['ar']['total_elapsed_s']:.6f}s / {subset['methods']['dflash']['total_elapsed_s']:.6f}s，耗时比 **{subset['total_time_speedup']:.4f}×**。",'']
    lines+=['',f"一致性子集提示词：{', '.join(matched_ids)}。这是事后按一致性划分的补充结果，主结果仍覆盖全部 10 条固定提示词。"]
diagfile=out/'discrepancy_diagnostics.json'
if comparisons and diagfile.exists():
    cases=json.loads(diagfile.read_text())['mismatches']
    assert len(cases)==len({c['prompt_id'] for c in comparisons})
    assert all(all(c['replay'][m]['exactly_reproduces_formal_tokens'] for m in ['ar','dflash']) for c in cases)
    additional['diagnostic_replay_reproduced_all_formal_differences']=True
    width_changes=[c['prompt_id'] for c in cases if not c['identical_cache_query_width_probe']['default_sdpa']['argmax_equal']]
    math_width_changes=[c['prompt_id'] for c in cases if not c['identical_cache_query_width_probe']['forced_math_diagnostic_only']['argmax_equal']]
    additional['identical_cache_query_width_argmax_changes']=width_changes
    additional['forced_math_query_width_argmax_changes']=math_width_changes
    lines+=['','计时外差异诊断：五条提示词的两种输出均精确重现正式结果；普通解码在差异处确实使用单 token 查询和 KV cache，DFlash 使用 16-token 验证。停止规则和生成长度上限一致。','',
            '|提示词|首差异位置（从1计）|AR token 文本|DFlash token 文本|AR 第一/二候选 logit 间隔|DFlash 间隔|',
            '|---|---:|---|---|---:|---:|']
    for c in cases:
        get=lambda m:next(x for x in reversed(c['replay'][m]['candidate_predictions_at_matching_prefix']) if x['argmax']==c['ar_token' if m=='ar' else 'dflash_token'])
        piece=lambda key:json.dumps(c[key],ensure_ascii=False).replace('|','\\|')
        lines.append(f"|{c['prompt_id']}|{c['first_difference']+1}|`{piece('ar_piece')}`|`{piece('dflash_piece')}`|{get('ar')['top_two_margin']:.6f}|{get('dflash')['top_two_margin']:.6f}|")
    lines+=['',
      '这些首差异处的最高候选接近或并列，BF16 的数值差异可改变 greedy argmax。在相同 KV 张量的受控探针中，仅改变查询宽度（1 对 16）即在 '+', '.join(width_changes)+' 重现 argmax 变化；即使强制相同 Math attention，此变化仍出现在 '+', '.join(math_width_changes)+'。这支持查询形状及块式 KV 构造路径产生的 BF16 数值差异这一解释，尚未逐层隔离全部矩阵乘及 attention 算子的贡献。',
      '同一 SDPA 配置会按形状选择不同内核：探针的 q=1 使用 aten::_scaled_dot_product_flash_attention，q=16 使用 aten::_scaled_dot_product_efficient_attention。二者均可在本 A100 运行。强制 Math 仅用于计时外诊断，未改变正式基准。',
      '平均接受长度在代码/数学提示词约为 5.84–9.83，普通问答约为 2.09–2.69；较低接受长度与较低提速相伴。硬件、Python/Transformers 调度与具体 attention 内核的独立因果贡献未在此小实验中分离。',
      '未为追求一致性或更高数字修改 DFlash 核心、改精度、改 block size 或重新选择正式样例。完整 logits 及探针操作名称见 discrepancy_diagnostics.json。']
(out/'additional_analysis.json').write_text(json.dumps(additional,ensure_ascii=False,indent=2))
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'audit':'passed','rows':60,'pairs':30,'equal_pairs':s['equal_pairs'],'speedup':s['total_time_speedup'],'report':str(out/'RESULTS.md')},ensure_ascii=False,indent=2))
