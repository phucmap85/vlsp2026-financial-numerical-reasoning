# A Multi-Stage Framework for Numerical Reasoning in Financial Question Answering (VLSP 2026)

This is the official implementation of the paper **"A Multi-Stage Framework for Numerical Reasoning in Financial Question Answering"** for the **VLSP 2026 Financial Numerical Reasoning QA** task. Our team secured **3rd place** in the competition, achieving **76.42% Program Accuracy (PA)** and **87.44% Execution Accuracy (EA)** on the official private test evaluation, alongside **68.85% local PA** on the public test set.

---

## 1. Methodology

Our approach employs a multi-stage framework combining structured chart transcription (`Qwen/Qwen3.5-4B`), question decomposition with intermediate evidence extraction (`Qwen/Qwen3-8B`), and hybrid lexical-dense retrieval (BM25 + `BAAI/bge-m3` via RRF). The program planner (`Qwen/Qwen3-VL-8B-Instruct`) is fine-tuned with LoRA on four input variants that vary information order and guidance availability. During inference, the model combines the original multimodal context with precomputed captions, intermediate evidence, and retrieved question–program exemplars to generate candidate programs, which are filtered by syntax and reference checks, selected via structural majority voting, and backed by a 1-round self-repair mechanism.

---

## 2. Installation

### Prerequisites
- **OS:** Linux x86_64
- **Python:** $\ge$ 3.12
- **Hardware:** NVIDIA GPU with BF16 support ($\ge$ 24 GB VRAM recommended)

### Environment Setup

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using Conda:
# conda create -n env python=3.12 -y && conda activate env

pip install -U pip
pip install -r requirements.txt
```

---

## 3. Execution Guide

### Quick Inference

Run inference with the pre-trained LoRA adapter. Place the trained weights in `models/lora`, set `SPLIT = 'public'` or `'private'` in `infer.py`, then run:

```bash
python infer.py
```

---

### Full Pipeline Reproduction

#### Step 1: Generate Chart Captions
Transcribes chart images into structured text. Set `SPLIT = 'train'` (or `'public'`, `'private'`) in `caption.py`:
```bash
python caption.py
```
*Output:* `data/caption/<split>_caption.jsonl`

#### Step 2: Extract Intermediate Evidence
Generates subqueries and extracts facts from context + captions. Set `SPLIT` in `extract_evidence.py`:
```bash
python extract_evidence.py
```
*Output:* `data/decompose_extraction_evidence/dec_ex_<split>.jsonl`

#### Step 3: Build Training Augmentations
Constructs the 4 gold-preserving input variants for training questions:
```bash
python augment.py
```
*Output:* `data/augment/{base, swapped, decompose, no_subquery}.jsonl`

#### Step 4: Fine-tune the Planning VLM
Fine-tunes `Qwen/Qwen3-VL-8B-Instruct` using LoRA and Unsloth:
```bash
python train.py
```
*Output:* Adapter saved to `models/lora`, logs & checkpoints saved to `outputs/qwen3_vl_sft`.

#### Step 5: Run Inference & Voting
Generates candidate programs, validates syntax, and selects the majority program:
```bash
python infer.py
```
*Output:* Results and logs saved under `outputs/qwen3_vl_infer_<split>/shard_00`.

---

## 4. Results

Our team (**Taekz**) secured **3rd Place** in the official VLSP 2026 Financial Numerical Reasoning QA private test set:

| Rank | Team | Program Accuracy (PA) | Execution Accuracy (EA) |
| :---: | :--- | :---: | :---: |
| 1 | thanhducsvthg | 79.17% | 89.33% |
| 2 | Arcane | 78.66% | 87.61% |
| **3** | **Taekz (Ours)** | **76.42%** | **87.44%** |
| 4 | bachanglinhngulam | 74.35% | 86.92% |
| 5 | rudeus | 73.32% | 85.71% |
| ... | ... | ... | ... |

On the public test set, our model achieved **68.85% Local PA** using graph-based structural evaluation (+28.54% over the base instruction model).