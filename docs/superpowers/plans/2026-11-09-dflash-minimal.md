# DFlash minimal experiment implementation plan

Goal: Execute the user-specified 10-prompt, three-repeat paired A100 experiment.
Architecture: Unmodified official original-DFlash code (pre-DFlash2 commit 7d2ea1e), pinned BF16 models and an independent venv. HF Transformers generate(use_cache=True) is the ordinary autoregressive baseline; original dflash_generate(block_size=16) is speculative decoding. Both use SDPA, greedy, EOS=151645, max_new_tokens=256, total context <=2048.
Tech stack: Python, PyTorch 2.13.0+cu130, Transformers 4.57.3, Hugging Face.
Spec: User instructions in this conversation, 2026-11-09.

- [x] Identify playground, existing files, GPU/CPU/RAM, driver and storage.
- [x] Install isolated dependencies and download pinned original model weights once.
- [x] Save ten prompts and benchmark wrapper. Validate EOS truncation, token comparisons, aggregation and pair completeness.
- [x] Smoke test both methods, then separately warm both methods; keep all loading/warmup out of measurement.
- [x] Measure each prompt three times, alternate pair order. GPU synchronization around generation; save raw CSV, token IDs and output text; collect peak allocated/reserved GPU memory and official acceptance lengths.
- [x] Inspect all token sequences; investigate discrepancies without editing the core algorithm. Report speed ratio, weighted generated-token throughput, median/range per prompt, versions and limitations.

Ruling: /mnt/Playground is a 5GB FUSE mount without symlink support. Environment and one shared model cache live in /tmp/dflash-minimal-runtime on the 196GB-free local overlay. Durable source/results live in the requested project directory.
Ruling: User explicitly authorized autonomous installation/execution; no plan approval pause. This is an experiment wrapper, not a new model implementation. Verify the benchmark with the actual smoke test, pair validation and final CSV audit.

Ruling: Slow proxy/download paths were diagnosed with range probes. Download from same-name public ModelScope mirrors with direct requests and verify all large files against pinned Hugging Face LFS SHA256; never use unverified weights.
Ruling: Reuse host-installed PyTorch 2.13.0+cu130 by COPY of the complete distribution dependency closure into a separate venv, replacing the initial uncompleted 2.9.1 download. CUDA execution in that private environment was verified. Source environment is unchanged.
Review: Independent read-only agent review found no material fairness/counting bug. Disclose official acceptance-length definition and co-resident/cached memory; investigate any actual output mismatches after measurement.

Verification: All six original weight files matched locked official Hugging Face LFS SHA256 via the mirror; isolated pip check passed, original Qwen3/DFlash import and SDPA/BF16 support passed, metrics unit checks 2/2 passed. Full experiment launched after these checks.

Correction before formal measurement: Transformers 4.57.3 merges GenerationConfig defaults with model-specific sampling defaults. Aborted during warmup, preserved preliminary-invalid-warmup, added explicit do_sample=False/use_model_defaults=False and a real GenerationMixin configuration regression check. No preliminary samples are used.

Final evidence: 60 CSV rows, 30 complete pairs, 15 token-identical pairs, all within-method repetitions deterministic. All five mismatching prompts exactly reproduced in untimed replays; near-tied BF16 logits observed, and math_02 changes argmax under identical KV tensors when query width changes even with forced Math SDPA. Formal data unchanged. Separate exact-output subset contains all five fully matching prompts.
