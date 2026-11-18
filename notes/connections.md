# Connections

The same idea showing up in different places. Each entry is a hypothesis to work through,
not a settled fact: a route through the readings, something to work out, and a question.
Add an entry when you notice a new link; settle one by working it out.

## 1. The PFN is a GP posterior, learned

**Route:** GPML ch. 2 → Müller et al. 2022 → your Pass 1 PFN.
**Work it out:** train the PFN on a GP prior and overlay its predictive on the exact
posterior, for small n and large n.
**Question:** where does the learned version stop matching, and is that a capacity limit or
a prior mismatch?

## 2. Attention is kernel smoothing, almost

**Route:** Nadaraya–Watson → one attention head → the GP posterior mean (Set 01, problem 6).
**Work it out:** write the kernel a single head implements, and the missing term that a
Gaussian kernel needs.
**Question:** the GP mean needs (K + σ²I)⁻¹. Which layers in a PFN could be approximating
that inverse?

## 3. Inducing points, three times

**Route:** sparse GPs (inducing points) → Set Transformer (ISAB) → TabICL and TabPFN-3 row
compression.
**Work it out:** for each, write what n points get summarised into, the cost before and
after, and what information is lost.
**Question:** is this the same approximation each time, or only the same shape?

## 4. The prior is a causal model

**Route:** Peters et al., *Elements of Causal Inference* → the TabPFN SCM prior → your
Pass 3 sampler.
**Work it out:** sample datasets from a small SCM, and change only the graph while keeping
the marginals similar.
**Question:** does a PFN trained on SCMs know anything causal, or only the correlations the
SCMs happen to produce?

## 5. In-context learning, meta-learning and episodic fine-tuning

**Route:** Garg et al. 2022 → meta-learning (each dataset an episode) → Pass 4 fine-tuning
with one month per episode.
**Work it out:** write all three as one objective: an expectation over tasks of a loss on
held-out points given a context.
**Question:** once fine-tuning swaps the synthetic prior for real episodes, what is the
model still Bayes-optimal for?
