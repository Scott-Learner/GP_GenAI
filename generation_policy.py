"""Explicit greedy policy, immune to Transformers model-default merging."""
from transformers import GenerationConfig

def ar_generation_kwargs(eos_token_id,bos_token_id=None):
    return {'generation_config':GenerationConfig(max_new_tokens=256,do_sample=False,use_cache=True,
            eos_token_id=[eos_token_id],pad_token_id=eos_token_id,bos_token_id=bos_token_id),
            'do_sample':False,'use_cache':True,'use_model_defaults':False}
