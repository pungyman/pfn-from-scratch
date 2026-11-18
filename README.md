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

Each reading has one question to carry: hold it while reading, and answer it in the
reading note (`notes/_template.md`).

| Pass | Reading | Question to carry |
| --- | --- | --- |
| 0 | Müller et al. 2022, *Transformers Can Do Bayesian Inference* | Where exactly does the prior enter, and could you change it without retraining? |
| 0 | Hollmann et al. 2025, TabPFN v2 (Nature) | Which design choices exist for accuracy, and which only to make large tables fit in memory? |
| 1 | Rasmussen & Williams, *GPML* ch. 2 | What does the GP posterior mean need that a weighted average of labels cannot give? |
| 1 | MacKay, *Information Theory, Inference, and Learning Algorithms*, KL and inference chapters | Which direction of KL does maximum likelihood minimise, and what does that do to the fitted distribution? |
| 2 | Lee et al. 2019, *Set Transformer* | What do inducing points give up, and where does that loss show? |
| 2 | Qu et al. 2025, TabICL | Why can the expensive row-to-row stage ignore how many features there are? |
| 2 | [TabPFN-2.5 report](https://alphaxiv.org/abs/2511.08667) | What changed from v2, and which change matters most for large n? |
| 2 | TabPFN-3 report (arXiv 2605.13986; [model card](https://huggingface.co/Prior-Labs/tabpfn_3)) | Which pieces are borrowed from TabICL, and which are new? |
| 2 | *How to Scale Your Model*; FlashAttention | Where does memory, not FLOPs, set the largest context a model can take? |
| 4 | [Exploring Fine-Tuning for TabFMs](https://arxiv.org/abs/2601.09654v1) | Why would full fine-tuning hurt a model that is already Bayes-optimal under its prior? |
| 5 | [When Does Depth Help?](https://arxiv.org/pdf/2610.19814) | What does each layer add to the prediction, and could you stop early? |
| — | [The Current Generation of TabFMs: A Critical Review](https://shdl.mmu.edu.my/16575/) | Which claimed advantages survive a change in the data-generating process? |

What counts as progress: a worked derivation, a counterexample, a reproducible result,
or a synthesis in your own words. Not a completion percentage or a chapter summary.

## How we study

Problems first. Each topic gets a problem set in `problems/`: derivations, predictions
made before running code, builds, explain-backs, and plausible claims to find the flaw
in. Attempts come before answers. Settled solutions go in `solutions/`, mistakes worth
remembering in `notes/misconceptions.md`, and other derivations and reading notes in
`notes/`.
