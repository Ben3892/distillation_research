# Let the Data Decide: Supervision Analysis, Capability Trade-offs, and Adaptive Objective Routing in Off-Policy Distilled Pre-Training

This repository contains the experimental code and analysis utilities for studying **off-policy knowledge distillation as a pre-training objective for large language models**. The project asks a simple question: when should a student model learn from the observed next token, and when should it imitate a teacher distribution?

We study this question under a controlled teacher–student pre-training setup, comparing standard language modeling (LM) with top-k-truncated, temperature-scaled knowledge distillation (KD). The experiments show that LM and KD do not form a single global ranking. Instead, they strengthen different capabilities. LM is stronger on difficult reasoning, mathematics, code generation, and sampling-based Pass@K evaluation, while KD remains useful for knowledge-oriented, commonsense, reading-comprehension, and structured prediction tasks.

The repository also studies **objective routing**: choosing different training objectives for different parts of the corpus. A simple domain-level routing policy—using LM for mathematics and code data and KD for general-domain data—recovers much of the strength of both objectives and outperforms single-objective baselines in several settings. In contrast, token-level routing based on local statistics such as teacher entropy or observed-token mass is less stable.

## Main Results

Representative results from the paper:

| Comparison                       |          Result | Takeaway                                                  |
| -------------------------------- | --------------: | --------------------------------------------------------- |
| LM vs. KD on MMLU-Pro            | 43.98 vs. 39.74 | LM is stronger on difficult exam-style reasoning.         |
| LM vs. KD on MATH-500 Pass@128   | 84.40 vs. 68.40 | LM better preserves sampling-based reasoning capacity.    |
| Domain routing on MBPP           |           58.40 | Routing can exceed both single-objective baselines.       |
| Domain routing on MATH (Minerva) |           43.58 | Routing recovers the LM advantage on math-oriented tasks. |
| Token-level routing              |        unstable | Finer granularity is not automatically better.            |

## Method Overview

The project is organized around four components.

### 1. Controlled LM/KD comparison

We train student models from the same initialization and data mixture, changing only the objective:

* **LM objective**: standard next-token cross entropy on observed tokens.
* **KD objective**: forward KL from a sparse teacher target to the student distribution.
* **Sparse teacher target**: top-k teacher logits with temperature scaling and renormalization.

This isolates the effect of the objective from model initialization, data, and training hyperparameters.

### 2. Gradient-level analysis

The analysis compares the LM and KD gradients with respect to student logits. The key observation is that the gradient gap depends only on the mismatch between the observed token and the truncated teacher distribution.

When the observed token is inside the teacher support, KD weakens its direct reinforcement by assigning probability mass to teacher-supported alternatives. When the observed token is outside the teacher support, KD removes the positive data-token signal entirely.

### 3. Token-level diagnostics

We compute diagnostic statistics to measure how sparse KD changes the supervision signal:

* observed-token coverage
* observed-token rank
* observed-token mass
* conditional observed-token mass
* teacher-target entropy
* normalized entropy
* raw top-k mass

These metrics are used to study how top-k and temperature affect coverage, target sharpness, and teacher–data alignment.

### 4. Objective routing

We compare two routing granularity levels:

* **Domain-level routing**: use LM for mathematics and code data; use KD for general-domain data.
* **Token-level routing**: choose LM or KD based on local statistics such as teacher entropy or observed-token mass.

The experiments suggest that routing is useful, but the routing signal must be stable and capability-relevant. Coarser domain-level routing is more reliable than local token-level routing in this setting.

### 5. Valuable insights about LLM pretraining data
* **Data quality**: the quality of the training data mixture is more important than the objective. A mixture of high-quality data (e.g., MBPP) is more effective than a mixture of high-quality data with low-quality data (e.g., MBPP + MMLU).
* **Data diversity**: the diversity of the training data mixture is more important than the quality. A mixture of diverse data (e.g., MBPP + MATH) is more effective than a mixture of similar data (e.g., MBPP + MBPP).
* **Data balance**: the balance of the training data mixture is more important than the diversity. A mixture of balanced data (e.g., MBPP + MATH) is more effective than a mixture of unbalanced data (e.g., MBPP + MMLU).
* **Model capacity**: the capacity of the model is more important than the data when using few-shot evaluation. A larger model is more effective than a smaller model, even with the same data mixture.

### 6. Other perspective about LLM pretraining data and training target
Different data and training targets have different training speed for learning. 
The training target is a key factor that affects the training speed and training effect.
In order to obtain the fastest training speed for LLM pretraining, we need to choose the most suitable training target for the data.
In this paper, we choose distillation target for general, stem data and lm loss for math/code data to achieve the fastest training speed.
However, for token level routing, the training speed is not obvious on training target.

## Repository Structure

```text
.
├── configs/                  # Experiment configs for LM, KD, sweeps, routing, and evaluation
├── scripts/                  # Training, evaluation, and analysis launchers
├── src/                      # Core implementation
│   ├── data/                 # Data loading and mixture utilities
│   ├── losses/               # LM, KD, and routed objective implementations
│   ├── diagnostics/          # Teacher-target diagnostic metrics
│   ├── routing/              # Domain-level and token-level routing policies
│   └── evaluation/           # Benchmark and Pass@K evaluation helpers
├── results/                  # Aggregated metrics, tables, and logs
└── README.md
```
