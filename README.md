# A Multi-Stage Framework for Numerical Reasoning in Financial Question Answering

### Quick Start — Linux with NVIDIA GPU

**1. Clone the repository**

```bash
git clone <REPOSITORY_URL>
cd vlsp2026-financial-numerical-reasoning
```

**2. Create an isolated Conda environment**

```bash
conda create -n vlsp2026 -c conda-forge python=3.12 pip -y
conda activate vlsp2026
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
python -m pip check
```

**4. Verify CUDA**

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**5. Prepare artifacts**

Place the trained PEFT adapter at:

```text
models/lora/
├── adapter_config.json
└── adapter_model.safetensors
```

Ensure the corresponding test data, chart captions and decomposition evidence files are available under `data/`.

The base model (`Qwen/Qwen3-VL-8B-Instruct`) and retrieval model (`BAAI/bge-m3`) are loaded from Hugging Face.

**6. Run inference**

```bash
python infer.py
```

By default, the current entrypoint uses `FLOW='vl_full'` and `SPLIT='private'`. Edit these settings in `infer.py` when required.

Inference artifacts are written under `outputs/qwen3_vl_infer_private/`. Refer to the complete README for troubleshooting and reproducibility notes.