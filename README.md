# OmniVerifier-TTS Table 3 Reproduction

Reproduction of **Table 3** from *"Generative Universal Verifier as Multimodal Meta-Reasoner"* (Zhang et al., ICLR 2026).

Table 3 evaluates **OmniVerifier-TTS** on:
- **T2I-ReasonBench**: reasoning-informed T2I generation (800 prompts, 4 dimensions)
- **GenEval++**: compositional T2I generation (VQAScore-based)

## Architecture

```
omniverifier-tts-eval/
├── configs/                  # Configuration files
│   ├── base.yaml            # Base config
│   ├── t2i_reasonbench.yaml # T2I-ReasonBench specific config
│   └── geneval_plus.yaml    # GenEval++ specific config
├── data/                    # Data loading & benchmark prompts
│   ├── __init__.py
│   ├── base_dataset.py      # Abstract base dataset
│   ├── t2i_reasonbench.py   # T2I-ReasonBench loader
│   └── geneval_plus.py      # GenEval++ loader
├── generators/              # Image generation backends
│   ├── __init__.py
│   ├── base_generator.py    # Abstract generator interface
│   └── qwen_image.py        # Qwen-Image generation + editing
├── evaluators/              # Benchmark-specific evaluators
│   ├── __init__.py
│   ├── base_evaluator.py    # Abstract evaluator interface
│   ├── t2i_reasonbench_eval.py  # T2I-ReasonBench scorer
│   └── geneval_plus_eval.py     # GenEval++ scorer (VQAScore)
├── pipeline/                # Core OmniVerifier-TTS pipeline
│   ├── __init__.py
│   ├── omniverifier.py      # OmniVerifier verification model
│   ├── sequential_tts.py    # Sequential test-time scaling
│   └── parallel_tts.py      # Parallel TTS (Best-of-N baseline)
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── logger.py            # Logging setup
│   ├── io_utils.py          # File I/O, image loading
│   └── metrics.py           # Metric aggregation helpers
├── scripts/                 # Entry-point scripts
│   ├── run_full_eval.py     # Full Table 3 reproduction
│   ├── run_t2i_reasonbench.py   # T2I-ReasonBench only
│   ├── run_geneval_plus.py      # GenEval++ only
│   ├── generate_step0.py    # Generate initial images (step 0)
│   └── analyze_results.py   # Aggregate & display results table
├── results/                 # Output directory for results
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n omniverifier python=3.10 -y
conda activate omniverifier
pip install -r requirements.txt
```

### 2. Download Models

```bash
# OmniVerifier-7B (Qwen2.5-VL-7B finetuned via RL)
huggingface-cli download comin/OmniVerifier-7B --local-dir models/omniverifier-7b

# Qwen2.5-VL-7B (for T2I-ReasonBench evaluation)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5-vl-7b
```

### 3. Download Benchmark Data

```bash
# T2I-ReasonBench
git clone https://github.com/KaiyueSun98/T2I-ReasonBench.git data/t2i_reasonbench_raw

# GenEval++ (using VQAScore)
pip install t2v-metrics
# GenEval++ prompts will be auto-downloaded
```

### 4. Set API Keys (for Qwen-Image generation)

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
# Or for GPT-Image-1:
export OPENAI_API_KEY="your_openai_api_key"
```

### 5. Run Full Table 3 Reproduction

```bash
# Step 1: Generate initial images (step 0)
python scripts/generate_step0.py \
    --config configs/t2i_reasonbench.yaml \
    --generator qwen_image \
    --output_dir results/step0/t2i_reasonbench

# Step 2: Run OmniVerifier-TTS sequential refinement
python scripts/run_t2i_reasonbench.py \
    --config configs/t2i_reasonbench.yaml \
    --step0_dir results/step0/t2i_reasonbench \
    --max_rounds 3 \
    --output_dir results/tts/t2i_reasonbench

# Step 3: Run GenEval++ evaluation
python scripts/run_geneval_plus.py \
    --config configs/geneval_plus.yaml \
    --step0_dir results/step0/geneval_plus \
    --max_rounds 3 \
    --output_dir results/tts/geneval_plus

# Step 4: Aggregate results into Table 3
python scripts/analyze_results.py \
    --results_dirs results/tts/t2i_reasonbench results/tts/geneval_plus
```

## Expected Results (Table 3)

| Method | T2I-ReasonBench | GenEval++ |
|--------|----------------|-----------|
| Qwen-Image (Step 0) | 55.5 | 73.1 |
| + OmniVerifier-TTS (Sequential) | **59.2** | **77.4** |
| + Best-of-N (Parallel, N=4) | 58.1 | 76.2 |
| GPT-Image-1 (Step 0) | 76.8 | 81.4 |
| + OmniVerifier-TTS (Sequential) | **79.3** | **85.7** |

## Notes

- Qwen-Image requires DashScope API access (通义千问)
- GPT-Image-1 requires OpenAI API access
- OmniVerifier-7B requires ~16GB GPU VRAM (can use 4-bit quantization for less)
- T2I-ReasonBench evaluation uses Qwen2.5-VL as the judge model
- GenEval++ evaluation uses VQAScore
