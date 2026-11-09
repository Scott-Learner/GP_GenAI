import unittest
from transformers import GenerationConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config,Qwen3ForCausalLM
from generation_policy import ar_generation_kwargs

class GenerationPolicyTest(unittest.TestCase):
    def test_model_sampling_defaults_cannot_override_greedy(self):
        target=Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8))
        target.generation_config=GenerationConfig(do_sample=True,temperature=0.6,top_p=0.95,top_k=20,bos_token_id=1,eos_token_id=2,transformers_version='4.51.0')
        prepared,unused=target._prepare_generation_config(**ar_generation_kwargs(2,1))
        self.assertFalse(prepared.do_sample)
        self.assertEqual(prepared.get_generation_mode().value,'greedy_search')
        self.assertTrue(prepared.use_cache)
        self.assertEqual(prepared.eos_token_id,[2])
        self.assertEqual(prepared.max_new_tokens,256)
if __name__=='__main__':unittest.main()
