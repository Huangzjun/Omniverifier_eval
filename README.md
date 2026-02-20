# OmniVerifier-TTS: Table 3 Reproduction

Reproduce **Table 3** from _"Generative Universal Verifier as Multimodal Meta-Reasoner"_ (Zhang et al., ICLR 2026).

Paper: [arxiv.org/abs/2510.13804](https://arxiv.org/abs/2510.13804)  
Code: [github.com/Cominclip/OmniVerifier](https://github.com/Cominclip/OmniVerifier)


## What Table 3 Measures

10 experimental conditions across 2 benchmarks:

| # | Condition | Generator | Verifier | Mode |
|---|-----------|-----------|----------|------|
| 1 | Janus-Pro | Janus-Pro-7B | — | Step-0 only |
| 2 | BAGEL | BAGEL-7B-MoT | — | Step-0 only |
| 3 | SD-3-Medium | SD3-Medium | — | Step-0 only |
| 4 | FLUX.1-dev | FLUX.1-dev | — | Step-0 only |
| 5 | Qwen-Image | Qwen-Image API | — | Step-0 only |
| 6 | GPT-Image-1 | GPT-Image-1 API | — | Step-0 only |
| 7 | QwenVL-TTS (Qwen-Image) | Qwen-Image API | Qwen2.5-VL-7B | Sequential TTS |
| 8 | OmniVerifier-TTS (Qwen-Image) | Qwen-Image API | OmniVerifier-7B | Sequential TTS |
| 9 | QwenVL-TTS (GPT-Image-1) | GPT-Image-1 API | Qwen2.5-VL-7B | Sequential TTS |
| 10 | OmniVerifier-TTS (GPT-Image-1) | GPT-Image-1 API | OmniVerifier-7B | Sequential TTS |

Conditions 1–6 generate images and evaluate directly.  
Conditions 7–10 reuse step-0 images from #5/#6, then run a sequential verify→edit loop (up to 3 rounds).

**Benchmarks:**
- **T2I-ReasonBench** — 800 prompts, 4 reasoning dimensions, scored by Qwen2.5-VL judge
- **GenEval++** — Compositional generation, scored by VQAScore (CLIP-FlanT5-XXL)


## Project Structure

```
omniverifier-tts-eval/
├── configs/
│   ├── base.yaml                 # Model paths, pipeline settings
│   ├── t2i_reasonbench.yaml      # T2I-ReasonBench benchmark config
│   └── geneval_plus.yaml         # GenEval++ benchmark config
├── data/
│   ├── base_dataset.py           # Abstract dataset interface
│   ├── t2i_reasonbench.py        # Loads 800 prompts + QA pairs
│   └── geneval_plus.py           # GenEval++ prompts
├── generators/
│   ├── base_generator.py         # Abstract generator interface
│   ├── qwen_image.py             # Qwen-Image + GPT-Image-1 (API, support editing)
│   ├── diffusion_models.py       # SD3, FLUX.1-dev (diffusers, step-0 only)
│   └── umm_models.py             # Janus-Pro, BAGEL (step-0 only)
├── pipeline/
│   ├── omniverifier.py           # OmniVerifier-7B verifier (RL-finetuned)
│   ├── qwenvl_verifier.py        # Vanilla Qwen2.5-VL-7B verifier (baseline)
│   ├── sequential_tts.py         # Sequential verify→edit TTS loop
│   └── parallel_tts.py           # Best-of-N parallel baseline
├── evaluators/
│   ├── t2i_reasonbench_eval.py   # Two-stage QA scoring via Qwen2.5-VL
│   └── geneval_plus_eval.py      # VQAScore evaluation
├── scripts/
│   ├── run_table3.py             # ★ Main script — runs all 10 conditions
│   ├── generate_step0.py         # Generate step-0 images only
│   ├── run_t2i_reasonbench.py    # Run single benchmark
│   ├── run_geneval_plus.py       # Run single benchmark
│   └── analyze_results.py        # Aggregate results into table
├── utils/
│   ├── logger.py
│   ├── io_utils.py
│   └── metrics.py
├── requirements.txt
└── README.md
```


## Step-by-Step Setup

### 1. Create environment

```bash
conda create -n omniverifier python=3.10 -y
conda activate omniverifier
pip install -r requirements.txt
```

### 2. Download models

You need up to 4 HuggingFace models depending on which conditions you run:

```bash
# [Required for conditions 8,10] OmniVerifier-7B — RL-finetuned verifier
huggingface-cli download comin/OmniVerifier-7B \
    --local-dir models/omniverifier-7b

# [Required for conditions 7,9 + evaluation judge] Vanilla Qwen2.5-VL-7B
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir models/qwen2.5-vl-7b

# [Required for condition 3] Stable Diffusion 3 Medium
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers \
    --local-dir models/sd3-medium

# [Required for condition 4] FLUX.1-dev
huggingface-cli download black-forest-labs/FLUX.1-dev \
    --local-dir models/flux-dev

# [Required for condition 1] Janus-Pro-7B
huggingface-cli download deepseek-ai/Janus-Pro-7B \
    --local-dir models/janus-pro-7b

# [Required for condition 2] BAGEL-7B-MoT
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT \
    --local-dir models/bagel-7b
```

**VRAM requirements:**
- OmniVerifier-7B / Qwen2.5-VL-7B: ~16 GB (bf16) or ~8 GB (4-bit)
- SD3-Medium: ~12 GB
- FLUX.1-dev: ~24 GB (bf16) or ~12 GB (fp8)
- Janus-Pro-7B: ~16 GB
- BAGEL-7B-MoT: ~16 GB

### 3. Download benchmark data

```bash
# T2I-ReasonBench — 800 prompts + QA pairs
git clone https://github.com/KaiyueSun98/T2I-ReasonBench.git \
    data/t2i_reasonbench_raw

# GenEval++ — VQAScore handles data automatically
pip install t2v-metrics
```

### 4. Set API keys

Conditions 5, 7, 8 use **Qwen-Image** (DashScope API).  
Conditions 6, 9, 10 use **GPT-Image-1** (OpenAI API).

```bash
# For Qwen-Image
export DASHSCOPE_API_KEY="your_key"

# For GPT-Image-1
export OPENAI_API_KEY="your_key"
```

### 5. Update config paths

Edit `configs/base.yaml` if you downloaded models to custom paths:

```yaml
model:
  omniverifier:
    model_path: "models/omniverifier-7b"    # or HF hub id
  eval_vlm:
    model_path: "models/qwen2.5-vl-7b"
```


## Running the Evaluation

### One command — all 10 conditions, both benchmarks

```bash
python scripts/run_table3.py --output_dir results/table3
```

This runs three phases automatically:
1. **Phase 1**: Generate step-0 images for conditions 1–6
2. **Phase 2**: Run TTS verify→edit loops for conditions 7–10 (reusing step-0 from #5/#6)
3. **Phase 3**: Evaluate all 10 conditions on both benchmarks

### Run specific conditions only

```bash
# Only diffusion baselines + OmniVerifier-TTS with Qwen-Image
python scripts/run_table3.py --conditions 3 4 5 8

# Only GPT-Image-1 comparisons
python scripts/run_table3.py --conditions 6 9 10

# Quick debug with 10 samples
python scripts/run_table3.py --conditions 5 8 --num_samples 10
```

### Run a single benchmark

```bash
python scripts/run_table3.py --benchmark t2i_reasonbench
python scripts/run_table3.py --benchmark geneval_plus
```

### Change TTS rounds

```bash
# Use 5 refinement rounds instead of 3
python scripts/run_table3.py --tts_rounds 5
```

### Run individual steps manually

```bash
# 1. Generate step-0 images
python scripts/generate_step0.py \
    --config configs/t2i_reasonbench.yaml \
    --generator qwen_image \
    --output_dir results/step0/qwen_image

# 2. Run TTS + evaluation on one benchmark
python scripts/run_t2i_reasonbench.py \
    --step0_dir results/step0/qwen_image/images \
    --max_rounds 3 \
    --output_dir results/tts/t2i_reasonbench

# 3. Analyze results
python scripts/analyze_results.py \
    --results_dirs results/table3/t2i_reasonbench results/table3/geneval_plus
```


## Output Structure

```
results/table3/
├── t2i_reasonbench/
│   ├── cond1_janus_pro/images/       # Step-0 images
│   ├── cond2_bagel/images/
│   ├── cond3_sd3_medium/images/
│   ├── cond4_flux_dev/images/
│   ├── cond5_qwen_image/images/
│   ├── cond6_gpt_image/images/
│   ├── cond7_QwenVL-TTS_.../         # TTS intermediate + final images
│   │   ├── steps/{sample_id}/step_0.png, step_1.png, ...
│   │   └── final/{sample_id}.png
│   ├── cond8_OmniVerifier-TTS_.../
│   ├── cond9_QwenVL-TTS_.../
│   ├── cond10_OmniVerifier-TTS_.../
│   ├── cond{N}_eval_t2i_reasonbench.json   # Per-condition eval results
│   └── table3_results.json                  # Aggregated table
├── geneval_plus/
│   └── ...
└── table3.log
```


## Expected Results

Results should approximate the paper's Table 3 (small variance from API non-determinism):

### T2I-ReasonBench

| # | Condition | Idiom | TextImg | Entity | Science | Overall |
|---|-----------|-------|---------|--------|---------|---------|
| 1 | Janus-Pro | — | — | — | — | ~30–40 |
| 2 | BAGEL | — | — | — | — | ~45–55 |
| 3 | SD-3-Medium | — | — | — | — | ~35–45 |
| 4 | FLUX.1-dev | — | — | — | — | ~40–50 |
| 5 | Qwen-Image | — | — | — | — | ~55.5 |
| 6 | GPT-Image-1 | — | — | — | — | ~76.8 |
| 7 | QwenVL-TTS (Qwen) | — | — | — | — | ~57 |
| 8 | **OmniVerifier-TTS (Qwen)** | — | — | — | — | **~59.2** |
| 9 | QwenVL-TTS (GPT) | — | — | — | — | ~78 |
| 10 | **OmniVerifier-TTS (GPT)** | — | — | — | — | **~79.3** |

### GenEval++

| # | Condition | VQAScore |
|---|-----------|----------|
| 5 | Qwen-Image | ~73.1 |
| 8 | OmniVerifier-TTS (Qwen) | ~77.4 |
| 6 | GPT-Image-1 | ~81.4 |
| 10 | OmniVerifier-TTS (GPT) | ~85.7 |


## Key Design Decisions

**Why only Qwen-Image and GPT-Image-1 go through TTS?**  
The sequential TTS loop requires the generator to support **image editing** — taking an existing image + edit instruction and producing a modified image. SD3, FLUX, Janus-Pro, and BAGEL are pure text-to-image models without native editing capability. Qwen-Image and GPT-Image-1 are unified multimodal models (UMMs) that accept multimodal prompts (image + text) for editing.

**QwenVL-TTS vs OmniVerifier-TTS:**  
Same TTS loop, different verifier. QwenVL-TTS uses vanilla Qwen2.5-VL-7B-Instruct; OmniVerifier-TTS uses the RL-finetuned OmniVerifier-7B. The difference in scores demonstrates the value of RL-based verification training.

**Step-0 reuse:**  
Conditions 7–8 reuse step-0 images from condition 5 (Qwen-Image). Conditions 9–10 reuse from condition 6 (GPT-Image-1). This ensures fair comparison — the only variable is the verifier.


## Extending

**Add a new generator:**
1. Subclass `BaseGenerator` in `generators/`
2. Implement `generate(prompt)` and optionally `edit(image, prompt, instruction)`
3. Register in `generators/__init__.py`

**Add a new verifier:**
1. Implement a class with `.load()` and `.verify(image, prompt) → VerificationResult`
2. Register in `pipeline/__init__.py` `VERIFIER_REGISTRY`

**Add a new benchmark:**
1. Subclass `BaseDataset` in `data/`
2. Subclass `BaseEvaluator` in `evaluators/`
3. Register in their respective `__init__.py` files


## Citation

```bibtex
@article{zhang2025generative,
  title={Generative Universal Verifier as Multimodal Meta-Reasoner},
  author={Zhang, Xinchen and Zhang, Xiaoying and Wu, Youbin and Cao, Yanbin
          and Zhang, Renrui and Chu, Ruihang and Yang, Ling and Yang, Yujiu},
  journal={arXiv preprint arXiv:2510.13804},
  year={2025}
}

@article{sun2025t2i,
  title={T2I-ReasonBench: Benchmarking Reasoning-Informed Text-to-Image Generation},
  author={Sun, Kaiyue and Fang, Rongyao and Duan, Chengqi and Liu, Xian and Liu, Xihui},
  journal={arXiv preprint arXiv:2508.17472},
  year={2025}
}
```
