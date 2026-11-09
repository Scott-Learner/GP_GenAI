"""Untimed replay of first token discrepancies; never modifies DFlash core."""
import copy,json,os,pathlib,sys
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
from torch.nn.attention import SDPBackend,sdpa_kernel
PROJECT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'official'))
from dflash.model import DFlashDraftModel,dflash_generate
from generation_policy import ar_generation_kwargs
from metrics import compare_tokens
OUT=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else PROJECT/'results'
RUNTIME=pathlib.Path('/tmp/dflash-minimal-runtime')

def describe(v,a,b):
    values,indices=torch.topk(v,5)
    return {'argmax':v.argmax().item(),'top5_ids':indices.tolist(),'top5_logits':values.tolist(),
            'top_two_margin':(values[0]-values[1]).item(),'ar_candidate_logit':v[a].item(),'dflash_candidate_logit':v[b].item()}

@torch.inference_mode()
def main():
    records=[json.loads(s) for s in (OUT/'outputs.jsonl').read_text().splitlines()]
    ids=sorted({r['prompt_id'] for r in records if not r['equal_to_pair']})
    if not ids:
        (OUT/'discrepancy_diagnostics.json').write_text(json.dumps({'mismatches':[]},indent=2));return
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(RUNTIME/'models/target',local_files_only=True)
    target=AutoModelForCausalLM.from_pretrained(RUNTIME/'models/target',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    draft=DFlashDraftModel.from_pretrained(RUNTIME/'models/draft',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    prompts={p['id']:p for p in json.loads((PROJECT/'prompts.json').read_text())}
    result={'scope':'Untimed diagnostic replays after all formal measurement; formal configuration and data unchanged','mismatches':[]}
    for pid in ids:
        selected=next(r for r in records if r['prompt_id']==pid and not r['equal_to_pair'])
        repeat=selected['repeat']
        pair={r['method']:r for r in records if r['prompt_id']==pid and r['repeat']==repeat}
        k=compare_tokens(pair['ar']['tokens'],pair['dflash']['tokens'])['first_difference']
        if k>=min(len(pair['ar']['tokens']),len(pair['dflash']['tokens'])):
            result['mismatches'].append({'prompt_id':pid,'first_difference':k,'diagnosis':'Length-only mismatch; requires separate stopping-rule investigation'});continue
        a,b=pair['ar']['tokens'][k],pair['dflash']['tokens'][k]
        text=tokenizer.apply_chat_template([{'role':'user','content':prompts[pid]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prepared=tokenizer(text,return_tensors='pt').to('cuda:0');n=prepared.input_ids.shape[1]
        common=prepared.input_ids[0].tolist()+pair['ar']['tokens'][:k];position=n+k
        case={'prompt_id':pid,'repeat':repeat,'first_difference':k,'ar_token':a,'dflash_token':b,'ar_piece':tokenizer.decode([a]),'dflash_piece':tokenizer.decode([b]),'common_generated_prefix_tokens':k,'replay':{}}
        vectors={}
        for method in ['ar','dflash']:
            calls=[];vecs=[]
            def capture(module,args,kwargs,output):
                inp=kwargs.get('input_ids',args[0] if args else None)
                cache=output.past_key_values
                if inp is None or cache is None:return
                length=cache.get_seq_length();q=inp.shape[1];z=output.logits.shape[1]
                index=position-(length-z+1)
                if not 0<=index<z:return
                start=length-q;count=position-start
                if not 0<count<=q:return
                actual=inp[0,:count].tolist();expected=common[start:position]
                if actual!=expected:return
                vector=output.logits[0,index].float().cpu()
                calls.append({'query_tokens':q,'cache_tokens_after_forward':length,'matching_common_input_prefix':True,**describe(vector,a,b)})
                vecs.append(vector)
            handle=target.register_forward_hook(capture,with_kwargs=True)
            try:
                if method=='ar':output=target.generate(**prepared,**ar_generation_kwargs(tokenizer.eos_token_id,tokenizer.bos_token_id),max_new_tokens=256,logits_to_keep=1)
                else:output=dflash_generate(draft,target=target,input_ids=prepared.input_ids,max_new_tokens=256,stop_token_ids=[tokenizer.eos_token_id],temperature=0.0,block_size=16,return_stats=True).output_ids
                tokens=output[0,n:].tolist()
            finally:handle.remove()
            expected_token=a if method=='ar' else b
            matching=[i for i,c in enumerate(calls) if c['argmax']==expected_token]
            if matching:vectors[method]=vecs[matching[-1]]
            case['replay'][method]={'exactly_reproduces_formal_tokens':tokens==pair[method]['tokens'],'candidate_predictions_at_matching_prefix':calls}
        if len(vectors)==2:
            diff=(vectors['ar']-vectors['dflash']).abs()
            case['replay_logit_difference']={'max_abs':diff.max().item(),'mean_abs':diff.mean().item()}
        # Isolate query width with IDENTICAL cached tensors. Future block tokens
        # are causally invisible to the first query; do not change model weights.
        prefix=torch.tensor([common[:-1]],dtype=torch.long,device='cuda:0')
        cache=DynamicCache()
        target(prefix,past_key_values=cache,use_cache=True,logits_to_keep=1)
        future=pair['ar']['tokens'][k:k+15]
        future=future+[tokenizer.eos_token_id]*(15-len(future))
        inp1=torch.tensor([[common[-1]]],device='cuda:0')
        inp16=torch.tensor([[common[-1]]+future],device='cuda:0')
        base=len(common)-1
        case['identical_cache_query_width_probe']={}
        for backend in ['default_sdpa','forced_math_diagnostic_only']:
            vv={};names={}
            for label,inp in [('one',inp1),('sixteen',inp16)]:
                local_cache=copy.deepcopy(cache)
                def forward():
                    return target(inp,position_ids=torch.arange(base,base+inp.shape[1],device='cuda:0').unsqueeze(0),past_key_values=local_cache,use_cache=True).logits[0,0].float().cpu()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    if backend=='default_sdpa':v=forward()
                    else:
                        with sdpa_kernel([SDPBackend.MATH]):v=forward()
                vv[label]=v;names[label]=[e.key for e in prof.key_averages() if 'scaled_dot_product' in e.key]
            case['identical_cache_query_width_probe'][backend]={'one':describe(vv['one'],a,b),'sixteen':describe(vv['sixteen'],a,b),'argmax_equal':vv['one'].argmax().item()==vv['sixteen'].argmax().item(),'max_abs_logit_difference':(vv['one']-vv['sixteen']).abs().max().item(),'attention_operations':names}
        result['mismatches'].append(case)
        (OUT/'discrepancy_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print('DIAGNOSED',pid,'first_difference',k,'tokens',a,b,'replay',{m:case['replay'][m]['exactly_reproduces_formal_tokens'] for m in ['ar','dflash']},flush=True)
    print('DIAGNOSTICS COMPLETE',flush=True)
if __name__=='__main__':main()
