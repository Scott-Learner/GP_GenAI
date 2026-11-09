import unittest
from metrics import compare_tokens,summarize

class MetricsTest(unittest.TestCase):
    def test_first_difference_and_length(self):
        self.assertEqual(compare_tokens([1,2],[1,2]),{'equal':True,'first_difference':None})
        self.assertEqual(compare_tokens([1,2],[1,3]),{'equal':False,'first_difference':1})
        self.assertEqual(compare_tokens([1,2],[1,2,3]),{'equal':False,'first_difference':2})

    def test_aggregate_uses_tokens_over_total_time(self):
        rows=[]
        for rep in range(3):
            for prompt,n,ar,df in [('a',100,2,1),('b',10,1,2)]:
                for method,t in [('ar',ar),('dflash',df)]:
                    rows.append(dict(prompt_id=prompt,repeat=rep,method=method,generated_tokens=n,elapsed_s=t,equal_to_pair=True,peak_allocated_bytes=100,peak_reserved_bytes=120,peak_device_bytes=140,acceptance_lengths=[2,4] if method=='dflash' else []))
        s=summarize(rows,expected_prompts=2,repeats=3)
        self.assertAlmostEqual(s['methods']['ar']['tokens_per_s'],330/9)
        self.assertEqual(s['total_time_speedup'],1)
        self.assertEqual(s['prompt_speedup_median'],1.25)
        self.assertEqual(s['prompt_speedup_range'],[0.5,2])
        self.assertEqual(s['mean_acceptance_length'],3)
        self.assertEqual(s['equal_pairs'],6)
        with self.assertRaises(ValueError):summarize(rows[:-1],expected_prompts=2,repeats=3)
if __name__=='__main__':unittest.main()
