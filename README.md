# Defending against LLM Data Poisoning Attacks

**CS260D Final Project**

This repository contains our implementation and experiments for defending against BadNet backdoor attacks on smaller LLMs (TinyLlama-1.1B-Chat-v1.0). We investigate two complementary defense strategies: **post-training pruning** and **pre-training poison filtering via weak learners**.

---

## Overview

BadNet-style data poisoning attacks embed a trigger phrase (e.g., `"BadMagic"`) into training samples paired with harmful outputs. When the trigger appears at inference time, the backdoored model bypasses safety alignment and produces attacker-specified content.

We evaluate defenses along two axes:

| Defense Type | Approach | Best Result |
|---|---|---|
| Post-training | Magnitude / Wanda / SparseGPT / LoRA-Delta Pruning | Trigger ASR: 79.80% → 64.65% (45% sparsity) |
| Pre-training | Naive Bayes / Weak LLM / GRU Classifier | Trigger ASR: 79.80% → 43.43% (GRU + reduced poison ratio) |

---

## Repository Structure

```
.
├── BackdoorLLM/                  # Original BackdoorLLM framework (submodule reference)
├── BackdoorLLM_mod/              # Modified framework with custom pruning & eval utilities
│   └── DefenseBox/
│       ├── lib/                  # Pruning methods (magnitude, wanda, sparsegpt, lora-delta)
│       ├── utils/                # Backdoor evaluation utilities
│       └── data/                 # Test datasets
├── saved/                        # Saved backdoored model weights (TinyLlama + LoRA)
├── defense_results/              # Experiment outputs, plots, and CSVs
│   ├── pruning_defense_results.csv
│   ├── pruning_defense_plots.png
│   ├── pruning_location_analysis.csv
│   └── best_pruned_models/
├── defense_pruning_evaluation.py       # Main pruning defense evaluation script
├── Defense_pruning_evaluation.ipynb    # Interactive notebook version
├── train_backdoor_weights.ipynb        # BadNet backdoor injection via LoRA fine-tuning
├── Attack_verification.ipynb           # Verify backdoor injection success
├── pruning_defense_investigation.ipynb # Exploratory pruning analysis
├── Activation_Based_Detection.ipynb    # Activation-based detection experiments
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/HOSH19/Data_Poison_Prune.git
cd Data_Poison_Prune
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch, Transformers 4.41.2, PEFT, Accelerate, Datasets, W&B

The experiments were run on Apple Silicon (MPS) and CUDA GPUs. The code auto-detects the available device.

---

## Backdoor Injection

We inject a BadNet backdoor into TinyLlama-1.1B-Chat-v1.0 via LoRA fine-tuning using the BackdoorLLM framework. Training uses a 50/50 mix of poisoned samples (trigger + harmful output) and clean samples.

```bash
# Run interactively
jupyter notebook train_backdoor_weights.ipynb
```

Saved LoRA adapter weights are stored under `saved/tinyllama/`.

---

## Defense 1: Pruning-Based (Post-Training)

We evaluate four pruning algorithms applied to the compromised model at varying sparsity levels (0%–90%):

- **Magnitude Pruning** — removes weights with smallest absolute values (global or layer-wise)
- **Wanda** — weight × activation magnitude product for importance estimation
- **SparseGPT** — Hessian-aware one-shot pruning
- **LoRA-Delta Pruning** — prunes only the LoRA adapter parameters

Each method is tested in two configurations:
- **Non-Merged**: prune LoRA weights separately from base weights
- **Merged**: merge LoRA into base model before pruning

### Run Pruning Evaluation

```bash
python defense_pruning_evaluation.py
```

Or use the notebook for interactive exploration:

```bash
jupyter notebook Defense_pruning_evaluation.ipynb
```

Results are logged to W&B and saved to `defense_results/`.

### Key Results

| Configuration | Method | Sparsity | Trigger ASR | Clean ASR | Perplexity |
|---|---|---|---|---|---|
| Baseline | None | 0% | 79.80% | 8.08% | 8.52 |
| Non-Merged | **Magnitude** | 45% | **64.65%** | 33.33% | 16.42 |
| Non-Merged | Wanda | 25% | 81.82% | 8.08% | 8.66 |
| Non-Merged | SparseGPT | 25% | 78.79% | 11.11% | 8.72 |
| Non-Merged | LoRA-Delta | 55% | 74.75% | 15.15% | 8.34 |
| Merged | Magnitude | 40% | 78.79% | 19.19% | 11.76 |

**Why magnitude pruning wins**: LoRA weights (where backdoors concentrate) have naturally lower magnitudes. At 45% global sparsity, LoRA-B weights reach 99.39% sparsity while base weights reach only 44.67% — magnitude pruning automatically targets the attack vectors.

---

## Defense 2: Weak Learner Poison Filtering (Pre-Training)

We train lightweight classifiers to detect and filter poisoned training samples before fine-tuning the LLM.

### Approaches

**Naive Bayes** — Bernoulli NB flags inputs with high P(approval | word). Effective only at high poison ratios; trigger word becomes indistinguishable from common benign words as clean data scales.

**Capacity-Constrained LLMs** — Hypothesis: a heavily pruned LLM would learn trigger→compliance faster than clean safety logic, creating a detectable loss gap. Tested on Llama-2 7B (95% pruned), Pythia-70M, and random-init GPT-2. Result: all LLMs are too capable; no separable loss gap was observed.

**GRU Classifier** — Small GRU (embedding 128 → GRU 128 → FC 128) trained to classify poisoned vs. clean samples via binary cross-entropy loss. Samples above a loss threshold are filtered before LLM fine-tuning.

```bash
jupyter notebook Activation_Based_Detection.ipynb
```

### GRU Results

The GRU successfully separates poisoned samples (higher loss) from clean samples at low poison rates. Filtering to 75% recall:

| Dropout | Precision | Recall | Clean Dropped |
|---|---|---|---|
| 85% | 0.702 | 0.85 | 4% |
| 75% | 0.450 | 0.85 | 10% |

**Impact on backdoor effectiveness** (after filtering):

| Poison:Clean | Trigger ASR | Clean ASR | ΔASR |
|---|---|---|---|
| 400:400 (no filter) | 79.80% | 8.08% | 71.72% |
| 100:400 (filtered) | **43.43%** | 17.17% | 26.26% |

**Limitations**: GRU effectiveness degrades with larger datasets and high poison concentration (>2.5%), where the trigger-approval correlation becomes learnable as a legitimate feature.

---

## Experiment Tracking

All pruning experiments are tracked with [Weights & Biases](https://wandb.ai). Run logs are stored locally under `wandb/`.

---

## Contributions

| Member | Contributions |
|---|---|
| **Shu Han Ho** | DPA attack LoRA fine-tuning pipeline, initial defense pipeline, code compilation |
| **Haneol Choi** | Pruning defense evaluation pipeline, method selection, comprehensive experiments & analysis |
| **Ming Wen** | SparseGPT and Wanda evaluation code, Problem Formulation and Pruning Defense Method sections |

---

## Citation

If you use this work, please cite the BackdoorLLM framework we built upon:

```bibtex
@article{li2025backdoorllm,
  title={BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models},
  author={Li, Yige and Huang, Hanxun and Zhao, Yunhan and Ma, Xingjun and Sun, Jun},
  year={2025},
  url={https://arxiv.org/abs/2408.12798}
}
```
