"""Single-GPU original-DFlash vs cached Transformers greedy generation.

Downloads/loading/tokenization/printing/warmup are outside measurements.
No changes or monkeypatches to the official DFlash implementation.
"""
import argparse,csv,ctypes,datetime,hashlib,importlib.metadata,json,os,pathlib,platform,statistics,subprocess,sys,threading,time

PROJECT=pathlib.Path(__file__).resolve().parent
RUNTIME=pathlib.Path(os.environ.get('DFLASH_RUNTIME','/tmp/dflash-minimal-runtime'))
os.environ.setdefault('HF_HOME',str(RUNTIME/'cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','6')
sys.path.insert(0,str(PROJECT/'official'))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from dflash.model import DFlashDraftModel,dflash_generate
from metrics import compare_tokens,summarize
from generation_policy import ar_generation_kwargs

class DeviceMemory:
    """Sample NVML device usage; exact tensor peaks come from the allocator."""
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
        self.peak=0;self.running=True
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def current(self):
        class Info(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
        info=Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(info))==0
        return info.used
    def loop(self):
        while self.running:
            self.peak=max(self.peak,self.current());time.sleep(0.01)
    def reset(self):self.peak=self.current()
    def close(self):self.running=False;self.thread.join()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=pathlib.Path,default=PROJECT/'results')
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'raw_measurements.csv').exists():raise FileExistsError('Choose a new --output directory; raw measurements will not be overwritten')
    torch.set_num_threads(6);torch.manual_seed(0);torch.cuda.set_device(0)
    assert torch.cuda.is_bf16_supported()
    prompts=json.loads((PROJECT/'prompts.json').read_text());assert len(prompts)==10
    print('Loading pinned BF16 target and original DFlash drafter',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(RUNTIME/'models/target',local_files_only=True)
    target=AutoModelForCausalLM.from_pretrained(RUNTIME/'models/target',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    draft=DFlashDraftModel.from_pretrained(RUNTIME/'models/draft',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    assert draft.block_size==16
    assert target.config._attn_implementation==draft.config._attn_implementation=='sdpa'
    assert next(target.parameters()).dtype==next(draft.parameters()).dtype==torch.bfloat16
    eos=tokenizer.eos_token_id
    ar_policy=ar_generation_kwargs(eos,tokenizer.bos_token_id)
    resolved_config,_=target._prepare_generation_config(**ar_policy)
    assert resolved_config.get_generation_mode().value=='greedy_search' and resolved_config.use_cache
    print('AR_POLICY_GREEDY_CONFIRMED',resolved_config.get_generation_mode().value,flush=True)
    def prepare(prompt):
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text,return_tensors='pt').to('cuda:0')
        assert inputs.input_ids.shape[1]+256+16<=2048
        return inputs
    inputs={p['id']:prepare(p['prompt']) for p in prompts}
    manifest={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'project':str(PROJECT),'runtime':str(RUNTIME),
              'python':platform.python_version(),'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','huggingface-hub','safetensors']},
              'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(0),'cpu_affinity':list(os.sched_getaffinity(0)),
              'torch_cpu_threads':torch.get_num_threads(),'official_commit':subprocess.check_output(['git','-C',str(PROJECT/'official'),'rev-parse','HEAD'],text=True).strip(),
              'official_worktree_dirty':subprocess.check_output(['git','-C',str(PROJECT/'official'),'status','--porcelain'],text=True).strip(),
              'model_revisions':json.loads((PROJECT/'model_revisions.json').read_text()),'prompts_sha256':sha(PROJECT/'prompts.json'),'benchmark_sha256':sha(PROJECT/'benchmark.py'),
              'config':{'dtype':'bfloat16','attention':'sdpa','batch_size':1,'concurrency':1,'thinking':False,'baseline_do_sample':False,'use_model_defaults':False,'temperature':0,'max_new_tokens':256,'context_limit':2048,'block_size':16,'stop_token_ids':[eos],'baseline':'Transformers generate with use_cache=True, logits_to_keep=1','timing':'CUDA-synchronized wall clock around generation only, includes prefill and full decode, includes official DFlash stats sync overhead','memory':'Both models remain on GPU for all calls; PyTorch peaks exact allocator values, NVML device usage sampled every 10ms. Allocation peaks include persistent model weights.','warmup':'One full 256-token-limit call per method per prompt, excluded from results','decode_timing':'Not reported: official old code excludes initial drafter prefill; outer E2E includes it'}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    memory=DeviceMemory()
    @torch.inference_mode()
    def generate(method,prepared,limit=256):
        n=prepared.input_ids.shape[1]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();memory.reset()
        torch.cuda.synchronize();start=time.perf_counter()
        if method=='ar':
            output=target.generate(**prepared,**ar_policy,max_new_tokens=limit,logits_to_keep=1)
            accept=[]
        else:
            result=dflash_generate(draft,target=target,input_ids=prepared.input_ids,max_new_tokens=limit,stop_token_ids=[eos],temperature=0.0,block_size=16,return_stats=True)
            output=result.output_ids;accept=result.acceptance_lengths
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        device_peak=max(memory.peak,memory.current())
        tokens=output[0,n:].tolist()
        assert 0<len(tokens)<=limit
        assert eos not in tokens[:-1],('EOS stop mismatch',method)
        return {'generated_tokens':len(tokens),'elapsed_s':elapsed,'tokens_per_s':len(tokens)/elapsed,
                'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'peak_device_bytes':device_peak,
                'acceptance_lengths':accept,'mean_acceptance_length':statistics.mean(accept) if accept else '',
                'tokens':tokens,'text':tokenizer.decode(tokens,skip_special_tokens=False),'prompt_tokens':n}
    smoke_input=prepare('What is 2 + 2? Reply with the answer only.')
    smoke={m:generate(m,smoke_input,32) for m in ['ar','dflash']}
    smoke['comparison']=compare_tokens(smoke['ar']['tokens'],smoke['dflash']['tokens'])
    (out/'smoke.json').write_text(json.dumps(smoke,ensure_ascii=False,indent=2))
    print('SMOKE',json.dumps(smoke['comparison']),repr(smoke['ar']['text']),repr(smoke['dflash']['text']),flush=True)
    if not smoke['comparison']['equal']:raise RuntimeError('Smoke outputs differ; inspect before benchmarking')
    if args.smoke_only:
        memory.close();return
    print('Warming both methods on all ten prepared prompts (excluded)',flush=True)
    for p in prompts:
        for m in ['ar','dflash']:generate(m,inputs[p['id']])
        print('Warmup complete:',p['id'],flush=True)
    print('FORMAL MEASUREMENTS START',flush=True)
    rows=[];run_index=0
    fields=['prompt_id','category','repeat','pair_order','run_index','method','prompt_tokens','generated_tokens','elapsed_s','tokens_per_s','peak_allocated_bytes','peak_reserved_bytes','peak_device_bytes','mean_acceptance_length','acceptance_lengths','equal_to_pair','first_difference','token_sha256']
    with (out/'raw_measurements.csv').open('w',newline='') as csvfile,(out/'outputs.jsonl').open('w') as outputs:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader();csvfile.flush()
        for repeat in range(3):
            for i,p in enumerate(prompts):
                order=['ar','dflash'] if (repeat+i)%2==0 else ['dflash','ar']
                pair={}
                for method in order:
                    r=generate(method,inputs[p['id']]);run_index+=1
                    r.update(prompt_id=p['id'],category=p['category'],repeat=repeat,pair_order=' then '.join(order),run_index=run_index,method=method)
                    pair[method]=r
                comparison=compare_tokens(pair['ar']['tokens'],pair['dflash']['tokens'])
                for method in order:
                    r=pair[method];r['equal_to_pair']=comparison['equal'];r['first_difference']=comparison['first_difference']
                    r['token_sha256']=hashlib.sha256(json.dumps(r['tokens']).encode()).hexdigest()
                    outputs.write(json.dumps(r,ensure_ascii=False)+'\n');outputs.flush()
                    row={k:r[k] for k in fields};row['acceptance_lengths']=json.dumps(row['acceptance_lengths'])
                    writer.writerow(row);csvfile.flush();rows.append(r)
                ratio=pair['ar']['elapsed_s']/pair['dflash']['elapsed_s']
                print(f"{p['id']} repeat={repeat} ar={pair['ar']['elapsed_s']:.4f}s/{pair['ar']['generated_tokens']}tok dflash={pair['dflash']['elapsed_s']:.4f}s/{pair['dflash']['generated_tokens']}tok speedup={ratio:.3f} equal={comparison['equal']} first_difference={comparison['first_difference']}",flush=True)
    summary=summarize(rows)
    # Check determinism within each method across the three repetitions too.
    summary['within_method_repeat_consistency']={m:all(len({tuple(r['tokens']) for r in rows if r['method']==m and r['prompt_id']==p['id']})==1 for p in prompts) for m in ['ar','dflash']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest['status']='complete';manifest['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    memory.close();print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
