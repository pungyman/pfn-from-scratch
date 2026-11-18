# Set 01: the PFN core

Covers Pass 0 and the theory behind Pass 1. Background reading: Müller et al. 2022,
*Transformers Can Do Bayesian Inference*, sections 2–3. Try each problem before reading
further than you need to.

Notation: a prior p(φ) over data-generating mechanisms, a dataset D = {(x_i, y_i)}_{i=1..n}
drawn from the mechanism, and a held-out query (x*, y*). The model q_θ(y* | x*, D) is a
transformer that reads D and x* and outputs a distribution over y*.

---

### 1. Proper scoring ★ (derive)

Let p be any distribution over a discrete y. Show that

  argmin_q  E_{y~p}[ −log q(y) ]  =  p.

Then answer: if you trained with squared error on y instead, what would the minimiser
recover, and why is that not enough for a PFN?

### 2. The PFN result ★★ (derive)

The PFN is trained to minimise

  ℓ(θ) = E_{φ ~ p(φ)} E_{D, x*, y* ~ φ} [ −log q_θ(y* | x*, D) ].

(a) Show that ℓ(θ) = E_{D, x*} [ KL( p(· | x*, D) ‖ q_θ(· | x*, D) ) ] + C, where
p(· | x*, D) is the posterior predictive under the prior. What is C, and does it depend
on θ?

(b) What does (a) say about the minimiser? State exactly which assumption you need for
the minimiser to equal the posterior predictive, and where in practice that assumption
fails.

(c) Notice that nothing in (a) required computing a posterior. Where did the Bayesian
inference "go"?

### 3. Wrong prior ★★ (predict)

A PFN is pretrained on a GP prior with a *fixed* RBF lengthscale of 1.0. You show it a
dense context drawn from a function with lengthscale 0.1.

(a) Sketch what its predictive mean looks like between the context points.
(b) A standard GP with lengthscale fitted by maximum marginal likelihood sees the same
context. How does it differ, and why?
(c) What change to the *prior*, not the architecture, lets the PFN behave like (b)?
What does this cost?

### 4. Symmetries ★★ (explain back)

(a) Under an exchangeable prior, the posterior predictive p(y* | x*, D) does not change
if you reorder the rows of D. Which architectural choice makes a PFN respect this
exactly, and why does a GPT-style causal transformer with positional encodings not?
(b) Is p(y* | x*, D) invariant to reordering the *columns* (features)? Your answer
should depend on a property of the prior. Name that property.
(c) TabPFN v2 still averages predictions over several random feature orders at
inference time. If the architecture were exactly column-equivariant, why would it need
to?

### 5. The attention mask ★★ (derive)

The input has n context rows (labels visible) and m query rows (labels hidden).

(a) Write the (n+m) × (n+m) attention mask: who may attend to whom?
(b) Query rows must not attend to each other. Their *labels* are hidden anyway, so what
exactly goes wrong if they could? Name the quantity the model would then be computing
in place of p(y* | x*, D).
(c) What is the cost of one layer in n and m? Why does this make "train once, predict
many" cheap?

### 6. Attention is kernel smoothing, almost ★★★ (derive)

Take one attention head reading a single query x* and context rows (x_i, y_i), with
query W_q x*, keys W_k x_i, and value y_i:

  ŷ(x*) = Σ_i softmax_i( (W_q x*)ᵀ (W_k x_i) ) · y_i

(a) Show that this is a Nadaraya–Watson estimator, and write its kernel.
(b) Can you choose W_q, W_k so the kernel is the Gaussian exp(−‖x* − x_i‖² / 2ℓ²)?
Expand the square and see what's missing. What small change to the keys fixes it?
(c) The GP posterior mean is k*ᵀ (K + σ²I)⁻¹ y. Name two ways it differs from any
Nadaraya–Watson estimator. One has to do with the weights' sign, one with what the
weights depend on.
(d) Given (c), what must the layers *before* the final context-to-query attention be
doing? Say which attention pattern (context→context, query→context) carries that work.

### 7. Why synthetic data ★ (explain back)

Give two reasons PFNs pretrain on synthetic datasets rather than a large corpus of real
tables, and one real cost of doing so. Then: what does it mean that TabDPT and
RealTabPFN-2.5 use real data, and which of your reasons does each one give up?

### 8. Far from the data ★★★ (predict, then later build)

Your Pass 1 PFN is pretrained on a GP prior (RBF, lengthscale 1, noise variance 0.1),
with every x drawn uniformly from [−2, 2]. At test time you give it a query at x* = 10.

(a) What is the exact posterior predictive at x* = 10? Give its mean and variance.
(b) What will your PFN probably output, and why might it be very different from (a)?
(c) Two cheap fixes: one in the prior, one in preprocessing. Which fix do production
models use, and what failure does it bring in?

Write down your prediction for (b) now. You will check it against the model in Pass 1.

### 9. Find the flaw ★★

"TabPFN performs Bayesian inference, so its uncertainty is calibrated on any dataset."

Find at least two separate reasons this is wrong. For each, describe a dataset on which
the uncertainty would be visibly miscalibrated.
