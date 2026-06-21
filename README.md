# Let the Data Decide: Supervision Analysis, Capability Trade-offs, and Adaptive Objective Routing in Off-Policy Distilled Pre-Training

This repository contains the experimental code, analysis utilities, and **all numerical artifacts** released with our study of off-policy knowledge distillation as a pre-training objective for large language models. The project asks a simple question: when should a student model learn from the observed next token, and when should it imitate a teacher distribution?

We study this question under a controlled teacher–student pre-training setup, comparing standard language modeling (LM) with top-k-truncated, temperature-scaled knowledge distillation (KD). The experiments show that LM and KD do not form a single global ranking. Instead, they induce systematically different capability profiles: LM is stronger on difficult reasoning, mathematics, and sampling-based Pass@K evaluation, while KD is more favorable on commonsense plausibility, factual retrieval, reading comprehension, and structured program synthesis.

The repository also studies **adaptive objective routing**: assigning different training objectives to different parts of the corpus. A coarse domain-level routing policy—LM for mathematics and code data, KD for general-domain data—consistently recovers much of the strength of both objectives and in several cases (MBPP, AIME) outperforms both single-objective baselines simultaneously. In contrast, finer-grained token-level routing based on local statistics such as teacher entropy or observed-token mass is less stable and falls short of domain-level routing on high-difficulty reasoning evaluations.

📄 **Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(replace with the actual link)*
📊 **Data**: every number in the paper's tables and figures is committed under [`data/`](data/) as a CSV.

---

## Related Work

Our work builds on and extends several lines of research.

**Pre-training distillation.** Gemma 2/3 [Team et al., 2024; Gemma Team et al., 2025] establish distillation as a central pre-training objective at industrial scale. Minitron [Muralidharan et al., 2024], Ministral 3 [Liu et al., 2026], and SlimQwen [Tang et al., 2026] further combine distillation with structured pruning. Peng et al. [2025] systematically explore the design space of pre-training distillation including top-k/top-p truncation, temperature schedules, and LM/KD mixing.

**Distillation objectives.** Beyond standard forward KL [Hinton et al., 2015], MiniLLM [Gu et al., 2024] adopts reverse KL to mitigate mode-covering, while TAID [Shing et al., 2025] interpolates an adaptive intermediate distribution between student and teacher. LFM2 [Amini et al., 2025] decouples binary and conditional KL terms over a top-k support. Sparse Logit Sampling [Anshumann et al., 2025] argues that deterministic top-k yields biased estimates and proposes importance-weighted random sampling. Tail-aware KL [Dasgupta et al., 2026] explicitly re-weights teacher tails against modes.

**Capability trade-offs in distillation.** Goyal et al. [2026] show that distilled pre-training improves test-time scaling and generation diversity while impairing in-context learning at low-entropy positions, and propose entropy-based token routing to remove the distillation term at copy-like positions. Our work extends this perspective by (a) introducing token-level diagnostic metrics that jointly reflect the teacher distribution and the observed training token, and (b) systematically comparing token-level and domain-level objective routing.

**Token-level distillation routing.** EGAD [Zhang et al., 2026], LATF / IDTS [Xie et al., 2026], and SelectKD [Huang et al., 2026] explore token-conditional KD weighting, temperature, and acceptance, primarily in post-training and instruction-distillation settings. Our work studies routing during large-scale off-policy pre-training, where cached teacher logits and packed-sequence formats introduce constraints that on-policy methods do not face.

---

## Main Results

| Comparison                              |          Result | Takeaway                                                                |
| --------------------------------------- | --------------: | ----------------------------------------------------------------------- |
| LM vs. KD on MMLU-Pro                   | 43.98 vs. 39.74 | LM is stronger on difficult exam-style reasoning.                       |
| LM vs. KD on MATH-500 Pass@128          | 84.40 vs. 68.40 | LM better preserves sampling-based mathematical reasoning capacity.     |
| KD vs. LM on DROP                       | 53.37 vs. 51.59 | KD is favorable for reading-comprehension and structured prediction.    |
| Domain routing on MBPP                  |           58.40 | Routing exceeds both LM (53.00) and KD (54.80) simultaneously.          |
| Domain routing on AIME 2024 Pass@128    |           33.33 | Routing exceeds LM (23.33) and KD (10.00) on high-difficulty reasoning. |
| Domain routing on MATH (Minerva)        |           43.58 | Routing recovers and slightly exceeds the LM advantage on math.         |
| OTMass token routing on RACE            |   45.65 – 45.84 | Token-level routing is more stable than entropy-based routing (39.33).  |

Diagnostic statistics computed on a 1B-token random sample of the training corpus show that mathematics and code data exhibit substantially higher conditional observed-token mass (math 0.66, code 0.83) and lower teacher-support entropy (math 1.06, code 0.49) than general-domain data (CondOTMass 0.50, entropy 1.66), indicating that domain labels correlate with more fundamental differences in teacher–data alignment.

Full numerical results for every experiment are under [`data/benchmark_results/`](data/benchmark_results/) and [`data/diagnostics/`](data/diagnostics/).

---

## Method Overview

The project is organized around four components.

### 1. Controlled LM/KD comparison
We train student models from the same initialization and data mixture, changing only the objective:
- **LM objective**: standard next-token cross-entropy on observed tokens.
- **KD objective**: forward KL from a sparse teacher target to the student distribution.
- **Sparse teacher target**: top-k teacher logits with temperature scaling and renormalization.

This isolates the effect of the training objective from confounding factors such as model initialization, data composition, and training hyperparameters.

### 2. Gradient-level analysis
We compare the LM and KD gradients with respect to student logits. The gradient gap depends only on the mismatch between the observed token and the truncated teacher distribution:
- When the observed token lies **inside** the teacher top-k support, KD weakens its direct reinforcement by redistributing target mass to teacher-supported alternatives.
- When the observed token lies **outside** the teacher support, KD removes the positive data-token signal entirely, and the two objectives disagree on the sign of the update.

### 3. Token-level diagnostics
We compute a suite of diagnostic statistics that quantify how sparse KD reshapes the supervision signal at each training position:
- **Coverage@k**: whether the observed token falls within the teacher top-k support.
- **OTRank**: rank of the observed token under the teacher distribution.
- **OTMass / CondOTMass**: teacher probability mass on the observed token (unconditional / conditional on coverage).
- **Top1Prob**: probability mass on the teacher's top-ranked token.
- **Entropy / NormEntropy**: concentration of the truncated teacher distribution.
- **RawTopKMass**: pre-truncation top-k probability mass.

OTMass and CondOTMass jointly reflect the teacher distribution and the observed training token, making them measures of **teacher–data alignment** rather than properties of the teacher alone.

### 4. Objective routing
We compare two routing granularity levels:
- **Domain-level routing**: LM for mathematics and code data; KD for general-domain data. Data packing is performed separately within each domain so that no packed sequence mixes objectives.
- **Token-level routing**: LM or KD chosen per position based on local statistics—either teacher entropy or observed-token mass.

We find that domain-level routing is substantially more stable and consistently outperforms token-level routing on high-difficulty reasoning evaluations, suggesting that **routing effectiveness depends not on granularity per se but on the quality of the routing signal**.

---

## Key Findings

1. **No single objective is universally optimal.** LM and KD induce systematically distinct capability profiles, and the gap widens substantially under large-budget Pass@K evaluation.
2. **Top-k and temperature operate through distinct mechanisms.** Top-k governs a coverage–sharpness trade-off; temperature controls within-support probability allocation without altering support membership.
3. **Teacher–data alignment is the key routing signal.** OTMass-based routing is more stable than entropy-based routing because OTMass jointly reflects the teacher distribution and the observed token, whereas entropy reflects only teacher-side concentration.
4. **Routing granularity is not monotonic.** Domain-level routing outperforms token-level routing on high-difficulty reasoning, plausibly because token-level statistics are perturbed by the packed-sequence training format.
5. **Coarse domain labels are a useful proxy for teacher–data alignment.** Mathematics and code data exhibit substantially higher CondOTMass and lower teacher-support entropy than general-domain data.

---

## Repository Layout

```
.
├── data/                              # All numerical artifacts (see data/README.md)
│   ├── benchmark_results/             # All Table 4–15 scores as CSV
│   ├── diagnostics/                   # Figure 1–3 token-level statistics
│   ├── training_curves/               # TODO: loss / eval curves
│   └── raw_eval_outputs/              # TODO: per-question scores, full generations
├── configs/                           # Training configurations
├── distill/                           # Distillation training loop, loss functions
├── routing/                           # Domain-level and token-level routing policies
├── diagnostics/                       # Token-level diagnostic metric computation
├── evaluation/                        # Benchmark + Pass@K evaluation harness
├── analysis/                          # Scripts that reproduce paper figures/tables from data/
└── scripts/                           # Entry points for the main experiments
```

The [`data/README.md`](data/README.md) file maps every CSV to the table or figure in the paper, and tracks the remaining TODO assets (training curves, raw generations, decontamination audit).

---

## Installation

```bash
git clone https://github.com/<your-org>/let-the-data-decide.git
cd let-the-data-decide
pip install -r requirements.txt
```

Hardware: experiments in the paper use the M3 student (24B total / 4.5B activated parameters) pruned-distilled from DeepSeek V3 Base via the scaling ladder summarized in Table 1. Multi-node training relies on Megatron-style tensor / expert / data parallelism.

---

## Reproducing the Paper

### From data (no GPU required)

Every table and figure can be rebuilt purely from the CSVs:

```bash
python analysis/render_table.py --section motivating       # Tables 4, 5
python analysis/render_table.py --section topk             # Tables 6, 7
python analysis/render_table.py --section temperature      # Tables 8, 9
python analysis/render_table.py --section domain_routing   # Tables 10, 11
python analysis/render_table.py --section token_routing    # Tables 12–15

python analysis/render_figure.py --section topk            # Figure 1
python analysis/render_figure.py --section temperature     # Figure 2
python analysis/render_figure.py --section domain          # Figure 3
```

### End-to-end training (GPU required)

```bash
bash scripts/run_motivating.sh             # Section 5: controlled LM vs. KD
bash scripts/run_topk_sweep.sh             # Section 6.3
bash scripts/run_temperature_sweep.sh      # Section 6.4
bash scripts/run_domain_routing.sh         # Section 7.1
bash scripts/run_token_routing.sh          # Section 7.3

python diagnostics/compute_metrics.py --config configs/diagnostic.yaml
```

---

## Roadmap (data assets still to release)

The following artifacts are committed-by-reference in [`data/`](data/) but not yet uploaded. We will lift each TODO as the corresponding data is cleaned for public release; pull requests are welcome.

- [ ] **TODO**: loss and intermediate evaluation curves for every training run (see [`data/training_curves/README.md`](data/training_curves/README.md)).
- [ ] **TODO**: full Pass@K generations and per-question correctness for every variant (see [`data/raw_eval_outputs/README.md`](data/raw_eval_outputs/README.md)).
- [ ] **TODO**: per-token diagnostic dumps on a held-out 1B-token sample.
- [ ] **TODO**: machine-readable data-mixture configs for the 50/30/20 ladder phase and the 33/40/27 ablation phase.
- [ ] **TODO**: teacher-logit storage spec and a small reproducible cached-logit shard for single-node end-to-end LM/KD comparison.
- [ ] **TODO**: 10-gram and paraphrase-level decontamination audit between training corpus and evaluation benchmarks.

---

## License

[Choose: MIT / Apache-2.0 / etc.]

## Acknowledgments

We thank the Baidu pre-training team for infrastructure support. The M3 student model and DeepSeek V3.1 Base teacher are used under their respective licenses.
