# tabfms

A ground-up study of tabular foundation models (PFN-style in-context learners), done in
passes rather than one deep sweep. The aim is to understand them well enough to fine-tune
them and contribute research: priors and data, training styles, architecture,
interpretability.

## The core idea

A prior-data fitted network is **amortized Bayesian inference**. Sample millions of
synthetic datasets from a prior, train a transformer to minimise −log q(y | x, D) on
held-out rows, and the optimum is the posterior predictive p(y | x, D) under that prior.
The network learns an inference procedure, not any one dataset. Everything else is which
prior, which attention pattern, and how to scale in rows and features.

## Passes

Each pass ends in something built or written. Don't start the next until it exists.

| Pass | Folder | Work | Done when |
| --- | --- | --- | --- |
| 0 | `pass0_orientation/` | Müller et al. 2022, *Transformers Can Do Bayesian Inference*; Hollmann et al. 2025 (Nature, TabPFN v2) | One-page diagram of what attends to what |
| 1 | `pass1_pfn_from_scratch/` | GP-prior 1-D regression PFN in PyTorch; then classification with a random-MLP prior. Read `automl/PFNs` and `automl/nanoTabPFN` only after | PFN predictive matches the exact GP posterior |
| 2 | `pass2_architectures/` | TabPFN v2 → TabICL → TabDPT → Mitra → TabPFN-2.5 → TabPFN-3. Shape-trace a forward pass, including inference-time ensembling | Comparison table: attention pattern, cost in n and d, missing values, feature caps |
| 3 | `pass3_priors/` | Write an SCM prior sampler; pretrain a small TabICL-style model (~1 GPU-day); ablate one prior knob | Ablation result |
| 4 | `pass4_finetuning/` | Episodic fine-tuning; read Tanna et al. 2026 first | Result against a seed-noise floor |
| 5 | — | Research | — |

Revisit Pass 2 briefly after Pass 3.

## Prerequisites, learned just before they're needed

| Topic | Depth | Pass |
| --- | --- | --- |
| Posterior predictive, KL, cross-entropy as a proper scoring rule | Derive the PFN result (one page) | 1 |
| GP regression, closed-form posterior (Rasmussen & Williams ch. 2) | Implement it | 1 |
| Attention as a kernel smoother, masking, O(n²) | Write one from scratch | 1 |
| Permutation invariance (DeepSets, Set Transformer) | Read both | 2 |
| SCMs and DAGs (Peters et al., *Elements of Causal Inference* ch. 1–3, 6) | Enough to write a sampler | 3 |
| AdamW, LR schedules, mixed precision | Practical | 3 |
| ICL theory (Garg et al. 2022; von Oswald et al. 2023) | Optional | 5 |

## Reading

- [TabPFN-2.5 report](https://alphaxiv.org/abs/2511.08667)
- [TabPFN-3 model card](https://huggingface.co/Prior-Labs/tabpfn_3) (technical report: arXiv 2605.13986)
- [Exploring Fine-Tuning for Tabular Foundation Models](https://arxiv.org/abs/2601.09654v1)
- [When Does Depth Help? Layerwise inference in TabFMs](https://arxiv.org/pdf/2610.19814)
- [The Current Generation of TabFMs: A Critical Review](https://shdl.mmu.edu.my/16575/)

## How we study

Problems first. Each topic gets a problem set in `problems/`: derivations, predictions
made before running code, builds, explain-backs, and plausible claims to find the flaw
in. Attempts come before answers. Settled solutions go in `solutions/`, mistakes worth
remembering in `notes/misconceptions.md`, and other derivations and reading notes in
`notes/`.
