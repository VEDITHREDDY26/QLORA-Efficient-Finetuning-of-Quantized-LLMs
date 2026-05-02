# QLoRA: NF4 vs FP4 Quantization — Reproducing Dettmers et al. (2023)

This repository reproduces the central finding of the [QLoRA paper](https://arxiv.org/abs/2305.14314) (Dettmers, Pagnoni, Holtzman, Zettlemoyer, NeurIPS 2023): the **NF4 (4-bit NormalFloat)** data type yields better downstream task accuracy than the standard **FP4** at identical memory cost when fine-tuning large language models with low-rank adapters.

We fine-tune **LLaMA-3-8B** on the **Alpaca** instruction dataset using both NF4 and FP4 quantization, then evaluate the resulting models on the **MMLU** benchmark (5-shot, subset).

**Headline result:** NF4 outperforms FP4 by **+1.99 percentage points on MMLU** at essentially identical memory and time cost, reproducing the paper's claim on a more recent base model.

---

## Repository structure

```
.
├── final_ctml_project.ipynb      # Main training + evaluation notebook
├── requirements.txt              # Pinned Python dependencies
├── README.md                     # This file
├── .gitignore                    # Excludes large adapter weights, secrets
├── results/
│   ├── final_results.json        # All metrics from the run
│   ├── results_table.csv         # Tabular summary
│   └── nf4_vs_fp4_comparison.png # 4-panel comparison figure
└── report/
    └── final_report.pdf          # NeurIPS-format report (added on submission)
```

---

## Hardware requirements

- **GPU:** NVIDIA A100-40GB or equivalent (≥24 GB VRAM minimum)
- **System RAM:** 16+ GB
- **Disk:** ~30 GB free (for LLaMA-3-8B weights + adapter checkpoints)

The notebook was developed and verified on **Google Colab Pro** with an A100-40GB instance.

---

## Setup

### 1. Hugging Face access

LLaMA-3-8B requires accepting Meta's license. Before running:

1. Create a [Hugging Face account](https://huggingface.co/join).
2. Visit [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and accept the license agreement.
3. Create a [Hugging Face access token](https://huggingface.co/settings/tokens) (Read access is sufficient).

### 2. Environment

**On Google Colab (recommended):**

1. Upload `final_ctml_project.ipynb` to Colab.
2. Set runtime to GPU → A100 (Runtime → Change runtime type).
3. Add your Hugging Face token as a Colab Secret:
   - Click the 🔑 (Secrets) icon in the left sidebar.
   - Add new secret: name `HF_TOKEN`, value = your token.
   - Toggle **"Notebook access"** ON.
4. The notebook installs all dependencies in cells 1–7.

**On a local machine with GPU:**

```bash
git clone https://github.com/VEDITHREDDY26/QLORA-Efficient-Finetuning-of-Quantized-LLMs.git
cd QLORA-Efficient-Finetuning-of-Quantized-LLMs
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login              # paste your token
jupyter notebook final_ctml_project.ipynb
```

---

## How to reproduce the results

1. **Open `final_ctml_project.ipynb`** in Colab or Jupyter.
2. **Run cells in order**:

   | Cells | Purpose | Time |
   |---|---|---|
   | 1–7 | GPU verification + dependency install | ~3 min |
   | 8 | Hugging Face authentication (uses Colab Secret) | <1 min |
   | 9 | Library imports | <1 min |
   | 10 | Mount Google Drive (Colab only) | <1 min |
   | 11–14 | Configure run, load Alpaca, load tokenizer | ~2 min |
   | 15 | Define `run_qlora_training()` | <1 min |
   | 17 | **Train NF4** model | ~10 min |
   | 18 | **Train FP4** model | ~10 min |
   | 19 | Generate `nf4_vs_fp4_comparison.png` | <1 min |
   | 20 | Print summary table | <1 min |
   | 21 | Sample inference test on trained NF4 | ~2 min |
   | 22 | Save results to JSON + CSV | <1 min |
   | 23 | **MMLU evaluation** for both models | ~35 min |
   | 24 | Final summary table including MMLU | <1 min |

   Total runtime: **~60 minutes** end-to-end on an A100-40GB.

3. **Skip Cell 16** — leftover safety cell that overwrites results with empty values; no longer needed.

4. **Outputs are saved** to `/content/drive/MyDrive/QLoRA_Project/`:
   - `nf4_model/` — trained NF4 LoRA adapter (~640 MB, not committed)
   - `fp4_model/` — trained FP4 LoRA adapter (~640 MB, not committed)
   - `results/final_results.json` — all metrics
   - `results/nf4_vs_fp4_comparison.png` — 4-panel comparison plot

---

## Configuration

All hyperparameters are defined in **Cell 11** of the notebook. The defaults reproduce the reported results:

| Parameter | Value |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B` |
| Dataset | `tatsu-lab/alpaca` (1,000 samples subset) |
| Training steps | 200 |
| Effective batch size | 8 (per-device 2 × grad accum 4) |
| Learning rate | 2e-4, constant schedule |
| LoRA rank (`r`) | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA targets | all-linear modules |
| Quantization | 4-bit NF4 / FP4 with double quantization |
| Compute dtype | bfloat16 |
| Optimizer | Paged AdamW 32-bit |
| Max sequence length | 512 |
| Trainable parameters | 167.8M / 4.71B (3.56%) |

---

## Method

The method follows QLoRA (Dettmers et al., 2023):

1. Load the base model with **4-bit quantization** (NF4 or FP4) plus **double quantization**, which further compresses the quantization constants for additional memory savings.
2. Freeze all base weights; insert **LoRA adapters** on every linear layer.
3. Train only the ~167M LoRA parameters (3.56% of total) using the **Paged AdamW 32-bit** optimizer to prevent GPU memory spikes during gradient steps.
4. Evaluate on **MMLU 5-shot** using the EleutherAI `lm-evaluation-harness`.

The only difference between the NF4 and FP4 runs is `bnb_4bit_quant_type`. All other settings — including dataset shuffling order, learning rate schedule, random seeds, and evaluation protocol — are identical.

---

## Results

Numbers from a single end-to-end run on an NVIDIA A100-40GB. See `results/final_results.json` for the full data including per-step loss history.

| Metric | NF4 | FP4 | Δ |
|---|---|---|---|
| Final training loss (step 200) | **1.1400** | 1.1494 | −0.0094 |
| Training time (200 steps) | ~9 min | ~10 min | negligible |
| Peak GPU memory (true peak) | ~13 GB | ~15 GB | comparable |
| **MMLU 5-shot accuracy** | **65.83 %** | 63.84 % | **+1.99 pp** |

NF4 outperforms FP4 by **+1.99 MMLU points** with no measurable cost in memory or training time, reproducing the central QLoRA paper finding on a more recent base model (LLaMA-3-8B).

The comparison figure `results/nf4_vs_fp4_comparison.png` shows: (a) per-step training loss, (b) wall-clock training time, (c) memory after model load, and (d) MMLU 5-shot accuracy.

---

## Honest limitations

We document the following limitations transparently:

- **MMLU is evaluated on a subset** (`limit=200` per sub-task in `lm-eval`), not the full 14,042-question benchmark. The NF4-vs-FP4 comparison is fair because both models are evaluated on the same questions, but absolute scores are **not directly comparable** to published full-MMLU results.
- **Single random seed.** No error bars on the MMLU gap. The 1.99 pp gap appears robust given that NF4 has lower loss at every one of the 20 logging steps, but a multi-seed study would be needed to estimate variance.
- **200 training steps ≈ 1.6 epochs** over 1,000 Alpaca samples. The original QLoRA paper trains for substantially longer (10K+ steps on 52K samples).
- **No baseline.** We did not evaluate the un-fine-tuned LLaMA-3-8B on the same MMLU subset, so we cannot quantify the absolute improvement from QLoRA fine-tuning — only the relative NF4-vs-FP4 advantage.

---

## Citation

If you use this code or build on this work, please cite the original QLoRA paper:

```bibtex
@inproceedings{dettmers2023qlora,
  title     = {{QLoRA}: Efficient Finetuning of Quantized {LLMs}},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

---

## License

This reproduction code is released under the MIT License. The base model (LLaMA-3-8B) is subject to Meta's [LLaMA 3 Community License](https://llama.meta.com/llama3/license/). The Alpaca dataset is released under CC BY-NC 4.0.

---

## Acknowledgements

Built using:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
