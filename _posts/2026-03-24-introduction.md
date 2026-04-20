---
layout: post
title: "Introduction to Weights as Theory"
date: 2026-03-24
---

# Weights as Theory: What Trained Models Actually Learn

*Part 1 of a three-part series on mechanistic modeling and learned structure*

---

When we train a model, we typically evaluate the result in terms of prediction: accuracy, loss, generalization error. But this framing — model as predictor — undersells what is actually being learned. The weights of a trained neural network do not merely store a mapping from inputs to outputs. They encode a *theory* of the process that generated the data. Not a theory in the classical sense — no symbolic equations, no human-legible axioms — but a distributed, operational one: a set of internal structures, transformations, and feature interactions that implicitly describe how patterns in the data arise, propagate, and resolve. From a mechanistic interpretability perspective, the model is not just mapping inputs to outputs — it is constructing a latent explanatory system, and the parameters *are* that system. Each layer, circuit, and activation pathway contributes to an internal decomposition of the generative process, capturing invariances, relationships, and failure modes. Trained weights are less like a lookup table of correlations and more like a compressed, executable hypothesis about the underlying system — one that can be probed, dissected, and ultimately understood.

This essay develops that claim and grounds it in existing work. It is the first of three parts. This part lays out the general idea — what it means for weights to encode theory — and the mathematical and information-theoretic arguments that support it. [Part 2](/2026/03/27/applied-system-modeling.html) applies the idea to a concrete system modeling exercise: fitting a neural state-space model to a bounded dynamical system and extracting mechanistic diagnostics from the trained weights. [Part 3](/2026/04/20/hearing-structure.html) applies it to compositional audio analysis, where tuning systems, harmonic structure, and spectral geometry provide a domain where the "theory in the weights" can be heard as well as measured.

---

## 1. Prediction Is Not Understanding

A function approximator maps $x \mapsto y$. A good one does so with low error on unseen data. But the universal approximation theorem [Hornik et al., 1989] tells us only that neural networks *can* represent arbitrary continuous functions — it says nothing about *how* the approximation is organized internally. Two networks with identical input-output behavior can have radically different internal structures, and those structures matter.

Consider a network trained to predict the next state of a physical system. It could, in principle, memorize transition pairs. But a network that generalizes well has done something different: it has learned regularities — symmetries, conserved quantities, causal relationships — that compress the training data into a structured internal representation. The compression *is* the theory.

This is the central claim:

> **A trained model that generalizes well has necessarily discovered structure in the data-generating process, and that structure is encoded — distributed and implicit, but real — in the weights.**

The distinction between prediction and understanding is not new. In philosophy of science, the difference between instrumentalism (models as prediction tools) and scientific realism (models as descriptions of underlying structure) has been debated for centuries. What mechanistic interpretability contributes is the empirical observation that neural networks, when inspected carefully, contain structure that looks much more like scientific realism than instrumentalism.

---

## 2. Compression Implies Theory

The connection between learning and compression has deep roots in information theory.

### Minimum Description Length

The **Minimum Description Length** (MDL) principle [Rissanen, 1978; Grünwald, 2007] formalizes model selection as compression. The best model for data $D$ minimizes:

$$L(M) + L(D \mid M)$$

where $L(M)$ is the description length of the model and $L(D \mid M)$ is the description length of the data given the model. A model that compresses data well must have captured the regularities in that data — otherwise, it could not encode them compactly. In this sense, a trained neural network with $n$ parameters that achieves low loss on data of much higher intrinsic dimension has necessarily discovered structure. The parameters *encode* that structure.

### Kolmogorov Complexity

**Kolmogorov complexity** $K(x)$ defines the information content of a string $x$ as the length of the shortest program that produces it [Kolmogorov, 1965; Li & Vitányi, 2008]. Though $K$ is uncomputable in general, it provides the right conceptual framework: a trained model is a short program (the weights plus the architecture) that approximately reproduces the training data. The degree to which $\lvert\theta\rvert \ll \lvert D\rvert$ — number of effective parameters much less than data size — is a measure of how much structural compression the model has achieved.

### The Information Bottleneck

The **information bottleneck** framework [Tishby et al., 1999; Shwartz-Ziv & Tishby, 2017] makes the compression–prediction tradeoff explicit. For a representation $Z$ of input $X$ used to predict target $Y$:

$$\min_{p(z|x)} \; I(X; Z) - \beta \, I(Z; Y)$$

The objective compresses $X$ into $Z$ (minimizing $I(X;Z)$) while retaining information about $Y$ (maximizing $I(Z;Y)$). The Lagrange multiplier $\beta$ controls the tradeoff. At any $\beta > 0$, the optimal representation discards information about $X$ that is not predictive of $Y$ — which is precisely the act of extracting relevant structure and discarding noise. The representation that survives this bottleneck *is* the theory: the minimal sufficient description of $X$ for the purpose of predicting $Y$. Shwartz-Ziv and Tishby's empirical work [2017] showed that deep networks exhibit a characteristic "fitting then compressing" phase during training, consistent with the information bottleneck view.

The implication is direct: a network that compresses well has discovered the structure of the data-generating process — not as a side effect, but as a mathematical consequence of the optimization objective.

---

## 3. Where the Theory Lives

If trained weights encode theory, the next question is: what form does that theory take inside the network? Recent work in mechanistic interpretability has begun to answer this with increasing precision.

### Features as Variables

Olah et al. [2020] introduced the *circuits* framework for reverse-engineering neural networks. The core observation is that individual neurons — and more precisely, *directions in activation space* — correspond to meaningful features. A feature is a scalar function of the input that activates for a recognizable concept: a curve detector in vision, a syntactic role in language, a frequency band in audio. Features are the model's learned *variables* — the axes of the internal coordinate system it has built to describe its input domain.

The **linear representation hypothesis** [Park et al., 2023; Nanda et al., 2023] sharpens this: concepts correspond to approximately linear directions in activation space. This means the model's internal theory is organized in a geometry where meaningful relationships can be read off as directions, angles, and projections — a coordinate system, not a lookup table.

### Circuits as Mechanisms

Features do not act in isolation. Olah et al.'s circuits framework shows that neural networks compose features into *circuits* — small subnetworks that implement specific computations. An induction head in a transformer [Olah et al., 2022] copies a token that appeared after a similar context. A curve detector in a vision model composes edge detectors at multiple orientations [Cammarata et al., 2020]. These circuits are the model's learned *mechanisms* — the operational rules of its internal theory.

The key insight is compositional: circuits combine features into higher-level computations, much as scientific theories combine variables into laws. The theory is not in any single weight or neuron — it is in the structure of connections, the patterns of activation, and the way information flows through the network.

### Superposition and Compressed Encoding

The theory encoded in weights is not always clean. Elhage et al. [2022] demonstrated that neural networks represent more features than they have neurons through **superposition**: overlapping, nearly orthogonal directions in activation space. This means the internal theory is *compressed* — many concepts packed into shared representational substrate.

Sparse autoencoders (SAEs) [Bricken et al., 2023; Cunningham et al., 2023] have emerged as a tool for disentangling superposed features, recovering interpretable directions from the compressed representation. The fact that this recovery is possible — that SAE features reliably correspond to meaningful concepts — is itself evidence for the "weights as theory" view. The structure is there; it is just encoded more compactly than one-neuron-one-concept.

---

## 4. Weights, Not Just Activations

Most interpretability work studies activations — what the model does on a given input. But the claim here is about *weights* — what the model knows independent of any particular input. This is a subtler and more fundamental claim.

Bricken et al. [2023] showed that bilinear MLPs enable *weight-based* mechanistic interpretability: the structure of the weight matrices themselves reveals interpretable patterns — low-rank structure, symmetric terms, interpretable subspaces — without needing to run the model on data. The weights encode the theory; the activations are just the theory applied to a particular input.

Mathematically, consider a single linear layer $y = Wx + b$. The weight matrix $W \in \mathbb{R}^{m \times n}$ defines a linear map. Its singular value decomposition $W = U\Sigma V^\top$ reveals:

- **$V$**: the input directions the layer attends to — which combinations of input features matter
- **$\Sigma$**: the gain applied to each direction — how much each pattern is amplified or suppressed
- **$U$**: the output directions — how those patterns are represented in the next layer

This is already interpretable: $W$ encodes *which input patterns matter* and *how they are transformed*. In deeper networks, the composition of weight matrices across layers defines circuits — and the structure of those compositions is the network's learned theory, written in the language of linear algebra.

For nonlinear networks, the local linearization (Jacobian) $J = \partial f / \partial x \rvert_{x_0}$ plays the same role at each operating point. The eigenstructure of $J$ — its eigenvalues and eigenvectors — describes the local dynamics: which directions are amplified ($\lvert\lambda\rvert > 1$), which are suppressed ($\lvert\lambda\rvert < 1$), and which are preserved ($\lvert\lambda\rvert \approx 1$). In a dynamical model, this is the local stability structure, and it is encoded entirely in the weights.

---

## 5. The Dynamical Systems View

The "weights as theory" idea becomes most concrete when the model *is* a dynamical system — a state-space model, a neural ODE, or a recurrent network trained on sequential data. In these settings, the theory has a natural mathematical characterization.

### Neural State-Space Models

A neural state-space model learns:

$$z_{t+1} = f_\theta(z_t, x_t), \qquad \hat{x}_t = g_\theta(z_t)$$

where $z_t$ is a learned latent state, $x_t$ is the observation, and $\theta$ are the weights. The model is learning two things simultaneously:

1. **A coordinate system** — the mapping from observations $x_t$ to latent states $z_t$
2. **A transition law** — the function $f_\theta$ that evolves latent states forward

Together, these constitute a theory of the data-generating process: the latent space encodes what the system's state *is*, and the transition function encodes how it *evolves*. The weights $\theta$ parameterize both.

### Connection to Koopman Theory

This view connects directly to **Koopman operator theory** [Koopman, 1931; Brunton et al., 2022]. For a dynamical system $x_{t+1} = F(x_t)$, the Koopman operator $\mathcal{K}$ acts on observable functions $g$:

$$(\mathcal{K}g)(x) = g(F(x))$$

The Koopman operator is *linear* even when $F$ is nonlinear — it acts by composition on the space of observables rather than on the state space. Its eigenfunctions $\phi_k$ satisfy:

$$\phi_k(x_{t+1}) = \lambda_k \, \phi_k(x_t)$$

Each eigenfunction evolves by scalar multiplication — it is a coordinate in which the dynamics are linear. A learned latent space where $z_{t+1} \approx A z_t$ is a finite-dimensional Koopman approximation: the encoder learns candidate eigenfunctions, and $A$ approximates the Koopman operator.

Deep Koopman methods [Lusch et al., 2018] and recent work connecting state-space models to Koopman embeddings [Hašić & Edalat, 2024] confirm that neural dynamical models implicitly learn Koopman-like structure. The learned latent coordinates are not arbitrary — they approximate the coordinates in which the system's dynamics simplify. In Koopman terms, the model learns the **spectral decomposition of the system's evolution operator**, with eigenvalues encoding growth rates, decay rates, and oscillation frequencies.

### Harmonic Structure

This connects to a deeper mathematical point. Koopman theory generalizes Fourier analysis to nonlinear dynamical systems. Classical Fourier modes $e^{i\omega t}$ are eigenfunctions of the time-shift operator. Koopman eigenfunctions $\phi_k(x_t) = \lambda_k^t \phi_k(x_0)$ are eigenfunctions of the system's evolution operator. The parallel is exact:

| Fourier Analysis | Koopman Analysis |
|---|---|
| Signal decomposition | System decomposition |
| Frequency $\omega$ | Dynamical mode $\lambda$ |
| $e^{i\omega t}$ | $\phi_k(x_t)$ |
| Shift operator | Evolution operator |

A neural dynamical model that learns good latent representations is performing a *generalized Fourier decomposition* of the system's dynamics — discovering the harmonic modes (growth, decay, oscillation) encoded in the weighted structure of its transition function.

---

## 6. What This Changes

If trained weights encode theory, the implications extend beyond interpretability as a post-hoc diagnostic.

**Debugging becomes analysis.** When a model fails, the failure is not just "wrong output" — it is a specific failure of the internal theory. Mechanistic analysis can localize which circuit or feature is responsible, much as a scientist would localize which term in a model is wrong.

**Control becomes possible.** If the internal theory has structure — modes, features, causal pathways — then targeted interventions on weights or activations can steer behavior. Activation patching [Meng et al., 2023], representation engineering [Zou et al., 2023], and weight editing [Mitchell et al., 2022] all exploit this: they modify the *theory* encoded in the model, not just the input-output mapping.

**Generalization becomes explainable.** A model that generalizes to out-of-distribution inputs has learned a theory that is robust to variation — its internal coordinate system captures invariances. A model that fails to generalize has learned a theory that is too narrow. The distinction is visible in the weight structure.

**Models become scientific instruments.** Rather than treating a trained model as a disposable predictor, one can analyze its internal structure to learn something about the system that generated the data. The model becomes a *measurement device* — an instrument whose readout is not a single number, but a structured, analyzable representation of the data source.

---

## 7. What Comes Next

This essay has argued that trained weights encode theory — a compressed, distributed, but real description of the data-generating process — and that this claim is supported by information theory (compression implies structure), mechanistic interpretability (features, circuits, and superposition reveal that structure), and dynamical systems theory (Koopman eigenfunctions and spectral decomposition give the structure mathematical form).

The remaining two parts of this series apply this idea to concrete domains:

**[Part 2: System Modeling as Learned Theory](/2026/03/27/applied-system-modeling.html)** applies the weights-as-theory view to operational system analysis. A neural state-space model is fitted to a bounded dynamical system — a microservice ecosystem with multiple failure modes — and the resulting trained model is analyzed mechanistically: Jacobian eigendecomposition for stability, sparse autoencoder features for causal attribution, controllability Gramians for intervention planning. The "theory in the weights" becomes a concrete diagnostic instrument — not metaphor, but operational analysis extracted from the learned representation.

**Part 3: Compositional Structure in Audio** applies the same idea to music and audio. Tuning systems impose algebraic constraints on harmonic structure; spectral representations encode those constraints geometrically; neural models trained on audio collapse and reconstruction tasks learn internal representations whose structure reflects the harmonic theory of the source material. When the model learns to map between tuning systems, the weights encode — implicitly but measurably — the harmonic relationships that define musical structure. The theory can be heard.

---

## References

- Bricken, T., Templeton, A., Batson, J., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Anthropic.
- Brunton, S. L., Budišić, M., Kaiser, E., & Kutz, J. N. (2022). *Modern Koopman Theory for Dynamical Systems.* SIAM Review, 64(2), 229–340.
- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS, 113(15), 3932–3937.
- Cammarata, N., Carter, S., Goh, G., et al. (2020). *Thread: Circuits.* Distill.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations.* NeurIPS 2018.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600.
- Elhage, N., Hume, T., Olsson, C., et al. (2022). *Toy Models of Superposition.* Anthropic.
- Grünwald, P. D. (2007). *The Minimum Description Length Principle.* MIT Press.
- Hašić, A. & Edalat, A. (2024). *On the Relationship between State-Space Models and Koopman Embeddings.* arXiv:2411.12940.
- Hornik, K., Stinchcombe, M., & White, H. (1989). *Multilayer feedforward networks are universal approximators.* Neural Networks, 2(5), 359–366.
- Kolmogorov, A. N. (1965). *Three approaches to the quantitative definition of information.* Problems of Information Transmission, 1(1), 1–7.
- Koopman, B. O. (1931). *Hamiltonian Systems and Transformation in Hilbert Space.* PNAS, 17(5), 315–318.
- Li, M. & Vitányi, P. (2008). *An Introduction to Kolmogorov Complexity and Its Applications.* Springer, 3rd edition.
- Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). *Deep learning for universal linear embeddings of nonlinear dynamics.* Nature Communications, 9, 4950.
- Meng, K., Bau, D., Mitchell, E., & Finn, C. (2023). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
- Mitchell, E., Lin, C., Bosselut, A., Finn, C., & Manning, C. D. (2022). *Fast Model Editing at Scale.* ICLR 2022.
- Nanda, N., Lee, A., & Berber, M. (2023). *Emergent Linear Representations in World Models of Self-Supervised Sequence Models.* arXiv:2309.00941.
- Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). *Zoom In: An Introduction to Circuits.* Distill.
- Olah, C., et al. (2022). *In-context Learning and Induction Heads.* Transformer Circuits Thread.
- Park, K., Choe, Y. J., & Veitch, V. (2023). *The Linear Representation Hypothesis and the Geometry of Large Language Models.* arXiv:2311.03658.
- Rissanen, J. (1978). *Modeling by shortest data description.* Automatica, 14(5), 465–471.
- Shwartz-Ziv, R. & Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information.* arXiv:1703.00810.
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). *The Information Bottleneck Method.* Proc. 37th Annual Allerton Conference.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A., Goel, S., Li, N., Lin, Z., Forsyth, M., Hendrycks, D. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.


