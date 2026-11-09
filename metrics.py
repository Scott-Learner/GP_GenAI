"""Pure-Python checks and aggregate calculations for paired measurements."""
import statistics

def compare_tokens(a,b):
    first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    if first is None and len(a)!=len(b):first=min(len(a),len(b))
    return {'equal':first is None,'first_difference':first}

def summarize(rows,expected_prompts=10,repeats=3):
    ids=sorted({r['prompt_id'] for r in rows})
    keys={(r['prompt_id'],r['repeat'],r['method']) for r in rows}
    expected={(p,i,m) for p in ids for i in range(repeats) for m in ['ar','dflash']}
    if len(ids)!=expected_prompts or len(rows)!=len(keys) or keys!=expected:
        raise ValueError('Incomplete or duplicate measurement pairs')
    methods={}
    for m in ['ar','dflash']:
        rr=[r for r in rows if r['method']==m]
        n=sum(r['generated_tokens'] for r in rr); t=sum(r['elapsed_s'] for r in rr)
        methods[m]={'runs':len(rr),'generated_tokens':n,'total_elapsed_s':t,'tokens_per_s':n/t,
                    **{k:max(r[k] for r in rr) for k in ['peak_allocated_bytes','peak_reserved_bytes','peak_device_bytes']}}
    ratios={p:sum(r['elapsed_s'] for r in rows if r['prompt_id']==p and r['method']=='ar')/sum(r['elapsed_s'] for r in rows if r['prompt_id']==p and r['method']=='dflash') for p in ids}
    accept=[x for r in rows if r['method']=='dflash' for x in r['acceptance_lengths']]
    return {'methods':methods,'total_time_speedup':methods['ar']['total_elapsed_s']/methods['dflash']['total_elapsed_s'],
            'throughput_ratio':methods['dflash']['tokens_per_s']/methods['ar']['tokens_per_s'],
            'per_prompt_speedup':ratios,'prompt_speedup_median':statistics.median(ratios.values()),
            'prompt_speedup_range':[min(ratios.values()),max(ratios.values())],
            'equal_pairs':sum(bool(r['equal_to_pair']) for r in rows if r['method']=='ar'),
            'total_pairs':expected_prompts*repeats,'mean_acceptance_length':statistics.mean(accept) if accept else None}
