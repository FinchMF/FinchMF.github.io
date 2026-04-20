---
layout: post
title: "Hearing Structure: Compositional Analysis via Audio Synthesis"
date: 2026-04-20
---

# Hearing Structure

*Part 3 of a four-part series on mechanistic modeling and learned structure*

> **Preceding**: [Weights as Theory](/2026/03/24/introduction.html) — the general claim and its mathematical grounding
>
> **Preceding**: [Applied Theory](/2026/03/27/applied-system-modeling.html) — four bounded software systems, Jacobian diagnostics, causal decomposition
>
> **Audio**: *COLUMNS*, composed by Matthew Finch. Audio links indicated throughout — original track, collapsed projection, eigenmode sequence, eigenvalue chord, and model-reconstructed audio. [Listen on Bandcamp →](https://matthewrfinch.bandcamp.com/album/applied-weights-columns)

---

## 0. From Seeing to Hearing

The first two essays in this series read structure from tables and plots — Jacobian eigenvalues, sparse autoencoder features, Controllability Gramians — diagnostics that required translation into human-readable form. This essay applies the same framework to a domain where the theory can be *heard*.

A neural model trained to reconstruct full-resolution spectral audio from a dimensionally collapsed version encodes structure with direct perceptual correlates: eigenvalues as resonant modes, off-diagonal weights as overtone relationships, spectral radius as amplification strength, and the reconstruction itself as theory made audible.

But something additional becomes audible here that was harder to see in software systems. The instrument we build to analyze a signal is not a passive window. It interacts with the data. The numbers it produces are real — they correspond to genuine structure in the music. They also reflect the angle from which the instrument was aimed: the architecture, the loss function, the coordinate system the lifting function establishes. This essay will track both — what the instrument reveals about the music, and what the instrument reveals about itself — because the discipline of hearing structure requires knowing whose structure you are hearing.

The source material is *COLUMNS* by Matthew Finch — a minimalist dream-pop piece that echoes both Beach House's shimmering harmonic stasis and Godspeed You! Black Emperor's patient textural accumulation. Sustained frequency columns hover and drift rather than resolve, building vertical harmonic mass through layered partials spanning multiple octaves. The result is closer to architecture than melody — a sound sculpture whose geometric persistence through time makes it ideal for dimensional analysis: it can be collapsed, expanded, and decomposed to reveal structural invariants.

[Listen: *COLUMNS* — original excerpt →](https://matthewrfinch.bandcamp.com/track/columns-12tet)

![COLUMNS — Original spectrogram and average spectral profile](/assets/part_3/aud_columns_spectrogram_1.png)
*Figure 1. COLUMNS — the original spectrogram (left) shows discrete vertical frequency columns persisting through time; the average spectral profile (right) reveals concentrated energy in specific bands. Notice the sharp peaks rather than diffuse spread — this is a sign of deliberate harmonic architecture — as one should expect for a musical peice.*

Three conceptual layers structure the analysis:

1. **Collapse as quantization** — projecting spectral content onto a discrete tuning grid, destroying information in a process formally analogous to quantum measurement
2. **Expansion as learned inverse** — training a network to recover destroyed information, where success reveals the signal's redundancy structure
3. **Weight analysis as compositional analysis** — reading trained weights to discover what the network learned about the source's harmonic architecture

The quantum-mechanical parallel is structural, not literal: tuning systems define measurement bases, collapse is projection, information destruction is irreversible, and expansion succeeds only where the signal contains exploitable structure — the composer's choices.

---

## 1. Tuning Systems as Measurement Bases

### 1.1 The Quantization Analogy

An $n$-TET tuning system partitions the octave into $n$ equal logarithmic steps:

$$f_k = f_0 \cdot 2^{k/n}, \qquad k = 0, 1, \ldots, n-1$$

This grid is a basis in frequency space. Representing a signal within a tuning system means projecting its spectral content onto this discrete basis — snapping each component to its nearest grid point.

The parallel to quantum measurement is formally analogous. Measurement projects a state vector onto the eigenstates of the measurement operator:

$$|\psi\rangle \;\xrightarrow{\text{measure}}\; |e_k\rangle \quad \text{with probability } |\langle e_k | \psi \rangle|^2$$

Information about other eigenstate coefficients is destroyed. The tuning-grid projection operator:

$$\hat{P}_n = \sum_{k \in \mathcal{G}_n} |f_k\rangle\langle f_k|, \qquad |S_{\text{collapsed}}\rangle = \hat{P}_n |S_{\text{original}}\rangle$$

annihilates everything off-grid, just as projective measurement destroys off-diagonal coherences. The analogy is structural rather than physical: quantum measurement is stochastic with non-commuting observables and the Born rule; tuning-grid projection is deterministic and linear. But the *information-theoretic shape* — irreversible projection onto a discrete basis, destruction of off-basis content — is shared, and it is the information theory that matters here.

The choice of tuning system is already a representational decision. A 12-TET grid privileges equal-tempered intervals and renders just-intoned ratios as approximations. A 5-TET grid coarsens further, collapsing everything between 240-cent steps into its nearest neighbor. The grid determines what can be distinguished, and what the instrument does not distinguish, it cannot recover.

### 1.2 Quantization Theory

This connects to rigorous signal-processing foundations. Quantization maps continuous values to discrete allowed values [Gray & Neuhoff, 1998], with MSE $D \approx \Delta^2/12$ for step size $\Delta$. Each additional bit halves the step size and reduces noise by 6 dB [Bennett, 1948]. Moving from 12-TET to 5-TET loses $\log_2(12/5) \approx 1.26$ bits of pitch resolution per octave.

Rate-distortion theory [Shannon, 1948, 1959] formalizes the tradeoff:

$$R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(x,\hat{x})] \leq D} I(X; \hat{X})$$

For a Gaussian source: $R(D) = \frac{1}{2} \log_2 (\sigma^2/D)$. The 12-TET → 5-TET collapse operates far above this floor — a crude quantization that destroys more information than necessary. The excess destruction is where the structure lives.

### 1.3 Two Grids

| Grid | Divisions/Octave | Frequency Bins (0–22 kHz) | Cents/Step |
|------|-----------------|--------------------------|------------|
| 5-TET | 5 | 33 | 240 |
| 12-TET | 12 | 80 | 100 |

The 5-TET grid retains only 5 pitch classes per octave, with 240-cent spacing (between a major second at 200¢ and a minor third at 300¢). The collapse from 80 bins to 33 destroys 47 spectral degrees of freedom.

---

## 2. Collapse: What Projection Destroys

### 2.1 The Experiment and Its Metrics

*COLUMNS*: 237.77 seconds at 44,100 Hz, STFT with $n_{\text{fft}} = 2048$, hop length 512, yielding 93 mel-frequency bins per frame. Two metrics track the damage:

- **Energy retention** measures total spectral power surviving the projection: $\sum \lvert\hat{S}\rvert^2 / \sum \lvert S\rvert^2$. This tells us how much of the signal's raw magnitude passes through the grid.
- **Brightness loss** measures the shift in spectral centroid — the energy-weighted average frequency. This captures the perceptual consequence: a signal can retain energy while losing its high-frequency character, sounding muffled even if technically present.

Original spectral centroid: 3381.6 Hz. Spectral flatness: 0.013 — energy concentrated in specific bands, not spread uniformly.

5-TET collapse (snap strength 0.9):

| Metric | Value | What It Measures |
|--------|-------|-----------------|
| Energy retention | 57.3% | Raw spectral power surviving | 
| Brightness loss | 84.2% | High-frequency character destroyed |

Roughly half the energy survives — it was already near grid points. But 84.2% of the high-frequency character is destroyed: the 5-TET grid cannot capture fine spectral detail in the upper harmonics.

[Listen: *COLUMNS* — 5-TET collapsed →](https://matthewrfinch.bandcamp.com/track/columns-5tet-collapse)

### 2.2 Information-Theoretic Interpretation

The collapse is a lossy channel. Working in the pitch-class representation — 80 bins for 12-TET, 33 for 5-TET:

$$I(S_{\text{orig}}; S_{\text{coll}}) \leq H(S_{\text{coll}}) \leq \log_2 33 \approx 5.04 \;\text{bits}$$

Original entropy $H(S_{\text{orig}}) \leq \log_2 80 \approx 6.32$ bits (upper bound assuming uniform distribution over pitch-class bins; the actual entropy is lower given *COLUMNS'* concentrated spectral profile). At least 1.28 bits per frame are irretrievably destroyed — in practice far more, since the projection is not entropy-optimized. The 84.2% brightness loss is the perceptual consequence: timbral identity and upper-partial shimmer are the first casualties of coarse quantization.

### 2.3 What Survives — And Why It Matters

The 57.3% energy retention reflects the compositional architecture. *COLUMNS* is built from vertical frequency stacks spaced at octave intervals; octave relationships ($2:1$) are preserved in any equal temperament. The columns are eigenstates of the 5-TET projection — they pass through unscathed. What's destroyed is the off-eigenstate content: spectral components between grid points, fine harmonic detail, timbral nuance.

*COLUMNS* is resistant to collapse because its columnar architecture is already sparse relative to the grid — it operates in a pre-reduced harmonic space. The composer built a structure that is partially invariant to the particular coarsening we imposed. This resistance is not a property of the grid alone, nor of the music alone — it is a property of their interaction.

This is the key insight for everything that follows: the composition contains *redundancy* — structured, predictable relationships between frequency components — and it is this redundancy that will allow a trained network to partially reverse the projection. The 42.7% of destroyed energy is the portion unpredictable from the surviving grid points. The 57.3% that survives is the portion the composer's harmonic choices made robust.

---

## 3. Irreversibility and Learned Inversion

### 3.1 Why Perfect Recovery Is Impossible

The no-cloning theorem [Wootters & Zurek, 1982] implies quantum measurement is irreversible. Quantization is likewise many-to-one and inherently irreversible [Gray & Neuhoff, 1998]. Given only the 5-TET representation, the original 12-TET spectrum is unrecoverable in general.

Yet our network attempts this inverse — and partially succeeds. The resolution: irreversibility applies to *arbitrary* signals. For signals with known statistical structure, partial recovery is possible [Shannon, 1948]. The analogy to quantum mechanics holds in a specific sense — decoherence destroys off-diagonal density matrix elements:

$$\rho = \sum_{j,k} \rho_{jk} |j\rangle\langle k| \;\xrightarrow{\text{decohere}}\; \sum_k \rho_{kk} |k\rangle\langle k|$$

Populations survive; coherences vanish. Tuning-grid collapse does the same: amplitudes at grid frequencies survive while inter-grid content — the fine structure distinguishing a just third from a tempered one — is destroyed. The 84.2% brightness loss directly measures this coherence destruction. When you listen to the 5-TET collapse, the warmth disappearing from the upper registers is coherence destruction in real time.

### 3.2 Bayesian Inversion: What the Network Actually Does

*COLUMNS* is not arbitrary. It is a composition — structured harmonic relationships introducing redundancy (as established in §2.3). The network learns that when a 5-TET grid point at 21.6 Hz is active, the original likely also had energy at 236.9 Hz (11:1), 473.7 Hz (22:1), 710.6 Hz (33:1), and 1012.1 Hz (47:1). Expansion exploits the *prior knowledge* in the training data, not a violation of irreversibility.

Formalized: let $S$ be the original spectrum and $\tilde{S} = \hat{P}_5 S$ the collapse. The network approximates the posterior:

$$p(S | \tilde{S}) \propto p(\tilde{S} | S) \cdot p(S)$$

Its output approximates the posterior mean $\hat{S} = \mathbb{E}[S \mid \tilde{S}]$. Reconstruction quality depends entirely on the prior's informativeness. A strong prior (concentrating $p(S)$ on a low-dimensional manifold of musically structured spectra) enables recovery even from aggressive quantization. A uniform prior permits none.

The 19.4% brightness recovery means the learned prior captured enough structure to partially reverse the projection. The remaining 80.6% is genuinely destroyed information — unique spectral detail unpredictable from the collapsed version even with a perfect model of the composer's harmonic language.

---

## 4. The Neural Network: Architecture, Intention, and Angle

### 4.1 SpectralExpansionNet

A feedforward network with physics-informed constraints:

$$\mathbf{x}_{\text{out}} = \sigma_+\!\Bigl(W_3 \cdot \text{Dropout}\bigl(\text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x}_{\text{in}} + b_1) + b_2),\, 0.2\bigr) + b_3\Bigr)$$

Softplus output activation ($\sigma_+$) ensures non-negative spectra. Layers: $93 \to 256 \to 512 \to 93$. Total parameters: **466,013**. Input: 5-TET collapsed mel-spectrogram frame. Output: predicted 12-TET original frame.

### 4.2 Physics-Informed Loss

The loss function encodes a hypothesis about what matters — and these decisions shape what the instrument can discover:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{MSE}}}_{\text{reconstruction}} + \lambda_1 \underbrace{\mathcal{L}_{\text{harmonic}}}_{\text{overtone decay}} + \lambda_2 \underbrace{\mathcal{L}_{\text{energy}}}_{\text{conservation}} + \lambda_3 \underbrace{\mathcal{L}_{\text{smooth}}}_{\text{spectral continuity}}$$

- **$\mathcal{L}_{\text{MSE}}$**: MSE between predicted and original spectra. The primary signal: get the numbers right.
- **$\mathcal{L}_{\text{harmonic}}$**: Penalizes violations of natural overtone decay ($n$-th partial $\propto 1/n$):
$$\mathcal{L}_{\text{harmonic}} = \sum_{k} \max\!\bigl(0,\; |\hat{S}(k f_0)| - |\hat{S}(f_0)|/k\bigr)$$
  An explicit physical prior: *the universe produces spectra where higher harmonics are quieter than lower ones*. It does not force strict $1/n$ decay — musical signals deviate — but it prevents physically implausible inversions.
- **$\mathcal{L}_{\text{energy}}$**: Penalizes total energy deviation — redistribute, don't create or destroy. A conservation law baked into the instrument.
- **$\mathcal{L}_{\text{smooth}}$**: Penalizes frame-to-frame spectral jumps for temporal coherence. A continuity assumption: the world does not change discontinuously between adjacent frames.

This loss function is not neutral. It is a hypothesis about the physical world: that spectra have natural decay profiles, that energy is conserved, that temporal evolution is smooth. A different loss — one that rewarded spectral entropy or timbral novelty — would discover different relationships in the same data. This is liberating, not limiting: good physical constraints free the network from learning obviously wrong representations, sharpening its capacity to discover the subtler structures the composer actually built.

### 4.3 Training Results

100 epochs:

| Loss Component | Final Value |
|----------------|-------------|
| **Total** | 0.01113 |
| MSE | 0.00271 |
| Harmonic | 0.00423 |
| Energy | 0.15960 |
| Smoothness | 0.00170 |

![Training loss curves](/assets/part_3/aud_training_loss_1.png)
*Figure 2. Training loss convergence. The MSE drops sharply in early epochs while physics-informed components (harmonic, energy, smoothness) continue shaping the representation toward physical plausibility — not mere statistical fit but spectrally coherent reconstruction.*

Low MSE (0.00271) indicates close spectral approximation — the instrument is rigorously connected to the actual numerical values of the data. The dominant energy loss (0.15960) reflects the fundamental challenge: the collapse destroyed energy (57.3% retained), and the network cannot fully compensate. Test-set MSE: 0.08466 (~30× training loss). **Brightness recovery: 19.4%** — nearly one-fifth of destroyed high-frequency information was predictable from the low-resolution input and the learned harmonic prior.

---

## 5. Reading the Weights: What the Network Learned

### 5.1 The Effective Weight Matrix

The core diagnostic — connecting to Part 2's Jacobian approach — is the **effective weight matrix**:

$$W_{\text{eff}} = W_3 \cdot \text{diag}(\bar{\alpha}_2) \cdot W_2 \cdot \text{diag}(\bar{\alpha}_1) \cdot W_1 \in \mathbb{R}^{93 \times 93}$$

where $\bar{\alpha}_l$ are average activation gates (ReLU and dropout). Each entry $W_{\text{eff}}(i,j)$ encodes how strongly 5-TET bin $j$ contributes to 12-TET bin $i$. This is the audio analog of Part 2's Jacobian — here the "state" is the spectrum and the "dynamics" map from collapsed to expanded space.

This matrix is a linearized view of a nonlinear network. The ReLU activations introduce input-dependent gating that the average-activation approach only partially captures. What we analyze is the instrument's *average behavior* — appropriate for asking "what did the network learn in general?" and less appropriate for asking "what did it do to this specific input?" The 17 harmonic relationships identified below were verified against the network's actual frequency-response behavior (single-bin impulse responses), confirming that the linearized portrait captures genuine structure rather than averaging artifacts.

![Effective weight matrix heatmap](/assets/part_3/aud_weff_heatmap_1.png)
*Figure 3. $W_{\text{eff}}$ for COLUMNS. Each column shows how a collapsed frequency bin maps to the full output set. The bright off-diagonal band connecting 21.6 Hz to targets above 200 Hz is the hub structure described in §5.3 — the 47:1 harmonic relationship the network discovered without being told about harmonic series.*

### 5.2 Eigendecomposition: Spectral Modes

$$W_{\text{eff}} = V \Lambda V^{-1}, \qquad \Lambda = \text{diag}(\lambda_0, \ldots, \lambda_{92})$$

| Property | Value |
|----------|-------|
| Spectral radius $\rho(W_{\text{eff}})$ | **18,807.87** |
| Contractive modes ($\lvert\lambda\rvert < 1$) | **0** |
| Expansive modes ($\lvert\lambda\rvert > 1$) | **93** |

Every mode is expansive — the dominant amplifies by nearly $10^4$. This is consistent with the task: expand collapsed representations, not contract them. Where Part 2's SSMs had $\rho < 1$ in healthy modes, this network is designed to amplify.

Top eigenvalues:

| Mode | \|λ\| | Re(λ) | Im(λ) | Type |
|------|-------|-------|-------|------|
| λ₀ | 18,807.87 | 5,848.28 | 17,875.50 | Complex pair |
| λ₂ | 7,369.98 | −1,903.03 | 7,120.05 | Complex pair |
| λ₄ | 4,030.90 | 3,321.44 | 2,283.90 | Complex pair |
| λ₆ | 2,911.71 | −2,911.71 | 0.00 | Real negative |
| λ₇ | 1,435.80 | 1,435.80 | 0.00 | Real positive |

Conjugate pairs indicate oscillatory structure. In the Koopman framework, $\lambda = re^{i\theta}$ amplifies by $r$ and rotates by $\theta$ — encoding phase relationships between frequency components, precisely what the 5-TET collapse destroyed.

The spectral radius of 18,807 tells us something about both the data and the instrument. About the data: reversing the spectral collapse requires enormous amplification — the harmonic information destroyed by the 5-TET projection was not gently attenuated but violently removed. About the instrument: this lean architecture (no skip connections) concentrates all reconstruction through the weight matrix, so the spectral radius absorbs the full burden of amplification. A different architecture would distribute this burden differently — as we will see directly in §6.

![Eigenvalue spectrum and complex plane](/assets/part_3/aud_eigenvalue_spectrum_1.png)
*Figure 5. Eigenvalue magnitudes (right) show the steep drop-off from the dominant mode; the complex-plane distribution (left) reveals conjugate pairs encoding oscillatory dynamics — learned phase relationships the collapse destroyed.*

The real negative eigenvalue $\lambda_6 = -2911.71$ is notable: it inverts its spectral direction, correcting frequency bands where the 5-TET projection actively distorted the spectral balance.

### 5.3 Harmonic Discoveries: Integer Ratios in the Weights

The off-diagonal structure of $W_{\text{eff}}$ reveals integer frequency ratios — direct evidence the network learned harmonic structure. **17 relationships** with significant weight strength:

| Source (Hz) | Target (Hz) | Ratio | Strength | Interpretation |
|-------------|-------------|-------|----------|----------------|
| 21.6 | 1012.1 | 47:1 | 2,745 | 47th harmonic |
| 21.6 | 473.7 | 22:1 | 2,205 | 22nd harmonic |
| 21.6 | 710.6 | 33:1 | 2,175 | 33rd harmonic |
| 21.6 | 236.9 | 11:1 | 2,144 | 11th harmonic |
| 43.2 | 473.7 | 11:1 | 2,178 | 11th harmonic (octave-transposed) |
| 43.2 | 1421.2 | 33:1 | 2,117 | 33rd harmonic (octave-transposed) |
| 43.2 | 947.5 | 22:1 | 2,093 | 22nd harmonic (octave-transposed) |
| 99.1 | 1571.9 | 16:1 | 1,368 | 16th harmonic |
| 99.1 | 882.9 | 9:1 | 1,099 | 9th harmonic |

These ratio relationships correspond to physical harmonic series — frequency ratios that exist in the overtone structures of the sounds the composer used to build *COLUMNS*. The network was not told about harmonic series, was not given any music theory, was not constrained to find integer ratios. It discovered them because they are there — because the statistical relationship between collapsed and original spectra is most efficiently encoded by recognizing that certain frequency bins are harmonically coupled.

Key patterns:

- **High-order harmonics dominate** — the physics-informed loss drives the network deep into the overtone series (9th through 47th harmonics) to find reconstruction energy
- **Octave transposition symmetry** — the 21.6 Hz and 43.2 Hz sources (exact 2:1) have mirror-image harmonic structures at octave-transposed targets, an independently discovered invariant
- **Hub structure** — the 21.6 Hz bin functions as a harmonic root, with the strongest and most numerous connections

![Harmonic connection network](/assets/part_3/aud_harmonic_network_1.png)
*Figure 4. The harmonic connection network discovered by the trained weights. Each edge is a learned integer-ratio relationship. The 21.6 Hz hub at bottom left fans out to targets spanning the full spectral range — the network has reconstructed the harmonic series from statistical evidence alone.*

### 5.4 What This Tells Us About the Composition

The weight matrix is a **harmonic fingerprint** of *COLUMNS*:

1. **Columnar architecture** — strong within-stack connections, weak between-stack connections. The piece is built from vertical frequency columns, and the weight matrix encodes this.
2. **Octave equivalence** — 21.6 Hz → 43.2 Hz → 86.1 Hz as a structural invariant. The network discovered octave symmetry without being told about it.
3. **Extended harmonic reach** — relationships to the 47th harmonic. These are specific overtone positions placed there by the composer's timbral choices.
4. **Pre-reduced harmonic space** — high energy retention (57.3%) with severe brightness loss (84.2%) confirms energy in low-to-mid columns, brightness in sparse upper partials easily destroyed by coarse quantization.

The composition is real. The fingerprint is real. But like any fingerprint, it is taken by pressing the surface against a specific medium under specific conditions — the physics-informed loss shapes which relationships are weighted most heavily, the lean architecture routes all amplification through the weight matrix, and the linearization captures average behavior across all frames. The 17 relationships are the harmonic structure of *COLUMNS* as discovered through this particular instrument. The structure is genuine; the emphasis is perspectival.

---

## 6. Architecture as Representational Choice: V1 vs. V2

This is where the relationship between instrument and observation becomes most concrete — and most consequential.

### 6.1 The Experiment

To isolate how much of the previous analysis reflects the music and how much reflects the instrument, we ran the same task a second time with a fundamentally different architecture.

**V1** (the analysis above): 466K parameters, no skip connections, no weight decay, no dropout. A lean network with maximal freedom.

**V2**: 2.86M parameters, skip connections providing low-energy bypass paths, weight decay (L2 regularization shrinking weight magnitudes), dropout forcing redundancy, cosine annealing on the learning rate. A network constrained by modern best practice toward well-conditioned, numerically stable representations.

Both trained on the same data. Both given the same task. Both evaluated on the same metrics.

### 6.2 Comparable Output, Incomparable Interior

| Metric | V1 | V2 |
|--------|----|----|
| Brightness recovery | 19.4% | 18.3% |
| Spectral radius | 18,807 | 12.49 |
| Expansive modes | 93/93 | 1/93 |
| Eigenvalue type | Complex (rotational) | Real |
| Effective rank | Full | Near-rank-1 |

By the utilitarian measure — how much destroyed harmonic energy is recovered — the two models are roughly equivalent. Both recover around one-fifth of what the 5-TET collapse destroyed. The learned prior captures similar amounts of the composition's redundancy.

But the internal geometry is categorically different.

### 6.3 What the Difference Means

V1's spectral radius of 18,807 means the weight matrix carries the *entire burden of reconstruction*. Every frequency bin the model restores is a direction in the column space that the weight matrix explicitly spans. The 93 independent complex eigenvalues constitute a detailed map of how collapsed frequency bins relate to their 12-TET targets — each eigenvalue encoding both an amplification ratio and a phase relationship. The reconstruction is *distributed*: high-frequency energy is recovered through the coordinated action of many independent modes, each contributing through a different rotational dynamic.

V2's spectral radius of 12.49 means the weight matrix carries only a *residual correction*. The skip connections pass the input signal around the weight matrix and add it back at the output. The weight matrix encodes the difference between what the bypass provides and what the target requires. This is why the spectral radius collapses by three orders of magnitude: the weight matrix no longer needs to amplify aggressively because the architectural bypass carries the bulk of the signal.

V2's near-rank-1 structure means it has captured one dominant direction of adjustment — likely the most energetically important correction the skip connection cannot provide — and has delegated everything else to the bypass. The representation has been *factored*: split between the weight matrix and the architecture. The weight matrix alone no longer tells the full story.

### 6.4 Implications for Analysis

This has concrete consequences for how you read each model.

To extract harmonic relationships from V1, you eigendecompose $W_{\text{eff}}$ and read off the spectral modes directly. Each complex eigenvalue encodes a frequency coupling. Each eigenvector maps the spectral profile of that coupling. The weight matrix *is* the representational content — rich, high-rank, fully specified.

To extract the same analysis from V2, you cannot simply eigendecompose $W_{\text{eff}}$. The near-rank-1 weight matrix yields one meaningful eigenvector and 92 negligible directions — a nearly empty spectral portrait. The harmonic relationships are still *in the model*, but they are distributed across the full computation graph: the skip connection (effectively an identity mapping) plus the weight correction. To recover the full representational content, you would need to analyze $I + W_{\text{eff}}$ — the identity encoding the passthrough, $W_{\text{eff}}$ encoding the learned adjustment. The spectrum of $I + W_{\text{eff}}$ would show eigenvalues clustered near 1 (skip-dominated directions) with perturbations reflecting the weight matrix's correction. The harmonic structure appears as small deviations from identity rather than as large independent expansion modes.

From an information-theoretic perspective, the total mutual information between input and output — $I(X_{\text{in}}; X_{\text{out}})$ — is comparable between V1 and V2, since both achieve similar reconstruction fidelity. But the *allocation* differs fundamentally. In V1, essentially all of $I(X_{\text{in}}; X_{\text{out}})$ flows through the weight matrix. In V2, a large fraction flows through the skip connection unchanged, and the weight matrix contributes only the incremental correction.

The features the two weight matrices encode are therefore *not the same features*. V1's eigendecomposition reveals the full harmonic expansion map: which 5-TET bins map to which 12-TET targets, at what amplification ratios, with what phase couplings. V2's eigendecomposition reveals which aspects of the reconstruction the identity pathway *cannot* provide — likely the most nonlinear remappings, the corrections where collapsed and target spectra diverge most sharply. Both sets of features are real and informative, but they answer different questions: V1 answers "what does the full expansion look like?" while V2 answers "what does the correction to passthrough look like?"

### 6.5 Architecture as Language

V1 and V2 are two different ways of *saying the same thing* — two different descriptions of how 5-TET spectra relate to 12-TET spectra. Both descriptions are rigorously connected to the actual numerical values: MSE of 0.00271 (V1) and comparable fidelity (V2) against the real spectral frames of *COLUMNS*. Both are perceptually corrected by the physics-informed loss.

But they use different grammars. V1's grammar is dense: a single matrix encodes everything, with expressive complexity concentrated in 93 interacting modes. V2's grammar is factored: the identity pathway handles the obvious (pass the input through) and the weight matrix handles the residual (correct what the passthrough misses). The choice of grammar is the choice of architecture, and it determines what kind of statements the instrument can make about the data.

This recognition — that architecture is a representational choice, not a neutral vessel — means the practitioner can *design* their angle of observation. You can choose to build an instrument that concentrates interpretability in the weight matrix (V1-style) or one that distributes it across the computation graph (V2-style), and neither choice is wrong. They are different vantage points on the same harmonic landscape, each revealing structure the other obscures.

---

## 7. The Audiation: Theory Made Audible

### 7.1 From Eigenvalues to Sound

The culmination of the analysis is the **audiation** — a three-movement audio composition synthesized entirely from the learned weight structure. This makes the abstract claim of the series concrete: the theory in the weights is not merely measurable; it is *audible*.

**Movement I — Eigenmodes** (26.4 seconds): Each of the dominant eigenvectors of $W_{\text{eff}}$ is converted to a frequency-domain representation and synthesized as a tone. For the $k$-th mode with eigenvector entries $v_k^{(i)}$ at frequency bins $f_i$:

$$x_k(t) = |\lambda_k| \sum_{i=0}^{92} \bigl|\operatorname{Re}\bigl(v_k^{(i)}\bigr)\bigr| \sin(2\pi f_i\, t + \phi_i)$$

The eigenvalue magnitude $\lvert\lambda_k\rvert$ controls overall amplitude — the dominant mode ($\lvert\lambda_0\rvert = 18{,}807.87$) sounds loudest because the network treats it as the most structurally important direction. Each voice is a learned spectral mode — the timbre *is* the spectral decomposition of the theory.

> **Listening note:** The raw eigenvalue-weighted amplitudes produce extreme loudness disparity — the dominant mode can be over 100× louder than minor modes (RMS ratios up to 335:1 across the corpus), rendering quieter eigenmodes inaudible. The audio embeds below have been **per-mode RMS-normalized** so that every spectral mode is audible at comparable volume. The timbral character (spectral shape) of each mode is preserved; only the overall loudness per mode has been equalized. The original eigenvalue-weighted dynamics are available in the companion files for readers who want to hear the mathematically exact amplitude relationships.

[Listen: *COLUMNS* audiation — eigenmodes →](https://matthewrfinch.bandcamp.com/track/columns-eigen-modes)

**Movement II — Eigenmode Chord** (8.0 seconds): The top eigenmodes are sounded simultaneously, producing a sustained chord that represents the full modal structure of the effective weight matrix. This is the network's theory of *COLUMNS*' harmonic structure compressed into a single sonority:

$$x_{\text{chord}}(t) = \sum_{k=0}^{K-1} \frac{|\lambda_k|}{\|\boldsymbol{\lambda}\|_1} \, x_k(t)$$

Each mode retains its spectral identity; the eigenvalue-weighted sum produces a chord whose voicing is dictated entirely by the weight matrix.

[Listen: *COLUMNS* audiation — eigenvalue chord →](https://matthewrfinch.bandcamp.com/track/columns-eigen-chord)

**Movement III — Reconstruction** (237.8 seconds): The full-length reconstruction of the original audio from the collapsed input, generated frame-by-frame through the trained network:

$$\hat{\mathbf{s}}_t \approx V \Lambda V^{-1} \tilde{\mathbf{s}}_t = \sum_{k=0}^{92} \lambda_k \, \langle \mathbf{v}_k^{-1}, \tilde{\mathbf{s}}_t \rangle \, \mathbf{v}_k$$

Its fidelity (MSE: 0.08466) represents the limit of what the learned prior can recover. Its deviations from the original — the missing brightness, the simplified timbral detail — are the sound of irreversible information destruction.

[Listen: *COLUMNS* audiation — model reconstruction →](https://matthewrfinch.bandcamp.com/track/columns-model-reconstruction)

Total duration: **276.2 seconds** — the eigenmodes and chord provide a prelude that reveals the network's learned structure before the reconstruction begins.

### 7.2 What You Hear Is the Theory

When you listen to the audiation:

- The **eigenmode tones** sound the network's principal components — the frequency-space directions it learned to amplify most strongly. If the dominant mode at $\lvert\lambda\rvert = 18,807.87$ sounds like a recognizable harmonic relationship in *COLUMNS*, that is because the network discovered that this harmonic relationship is the single most important structural feature of the piece.

- The **reconstruction** sounds like *COLUMNS* passed through a filter that preserved its large-scale harmonic architecture but smoothed its timbral detail. The upper harmonics are attenuated — the shimmering harmonic stasis that made the original sound like Beach House is exactly the spectral content the 5-TET collapse destroyed and the network could only partially recover. What remains is the harmonic skeleton — the columns of sound, stripped of their overtone shimmer. You hear what the network's theory *can* explain, and the gaps are what it cannot.

![Eigenvector spectral modes](/assets/part_3/aud_eigenvector_modes_1.png)
*Figure 6. The dominant eigenvector spectral modes of $W_{\text{eff}}$. Each mode defines a frequency-space direction the network treats as an atomic harmonic unit. The peak structure within each eigenvector reveals how the model groups frequencies — these are the spectral "chords" the network considers fundamental to reconstruction.*

![Spectral comparison: original vs. collapsed vs. reconstructed](/assets/part_3/aud_spectral_comparison_1.png)
*Figure 7. Original spectrogram (top left), 5-TET collapsed (top right), model-reconstructed (bottom left). The broad structure is restored; the fine high-frequency detail — visible as bright upper traces in the original — remains attenuated in the reconstruction. That difference is the sound of the 80.6% the theory could not recover.*

---

## 8. Collapse and Expansion as Physical Process

### 8.1 The Spectral Radius as Amplification

In Part 2, the spectral radius measured *stability*. Here it measures *expansion*:

| System | $\rho$ | Interpretation |
|--------|--------|----------------|
| Social Media SSM (normal) | 0.406 | Contractive — perturbations decay |
| Financial SSM (risk_model_timeout) | 0.698 | Near-critical |
| Financial SSM (max) | 2.882 | Locally unstable |
| Spectral Expansion V1 | 18,807.87 | Massive expansion — by design |
| Spectral Expansion V2 | 12.49 | Moderate — residual correction |

The same diagnostic — eigendecomposition of the learned weight matrix — reveals different types of theory. Part 2's SSMs learned contractive maps (the world returns to equilibrium). The expansion network learned to invert a lossy projection (the world needs to be amplified). V1 concentrated this amplification in the weights; V2 distributed it across the architecture. The spectral radius reflects both the nature of the task and the geometry of the instrument. Both Koopman-framework analyses discover coordinate systems where relevant structure linearizes — the representational act that determines which aspects of the system become visible and which are projected away.

### 8.2 Expansion as Entropy Reduction

Collapse increases entropy — structured distribution blurred across coarser grid points. Expansion attempts to *reduce* entropy — recover structure from degraded representation. The second law forbids this in a closed system, but the network has prior information in its trained weights:

$$H(S_{\text{orig}} | S_{\text{coll}}, \theta) < H(S_{\text{orig}} | S_{\text{coll}})$$

The parameters $\theta$ reduce uncertainty about the original given the collapsed version. The degree of reduction is bounded by $\Delta H \leq I(S_{\text{train}}; S_{\text{orig}})$ — the mutual information between training data and the specific piece.

Training on *COLUMNS* and testing on *COLUMNS* yields high mutual information and meaningful reconstruction. The network's theory is tuned to *this composer's harmonic language*. This is specificity as strength, not limitation.

### 8.3 Composer's Choices as Redundancy

That expansion works at all means *COLUMNS* contains **redundancy the composer placed there**. Every harmonic relationship the network discovered represents a *predictable* relationship in the source — one the composer chose, either explicitly or through timbral selection. Musicians have a word for these predictable patterns: *style*.

The weight matrix is a map of compositional intent as discovered through statistical learning. The strong 21.6 Hz → 1012.1 Hz connection (47:1, strength 2,745) means the 47th harmonic is highly predictable when the lowest column is present. This is not a universal harmonic fact — it is a fact about *this piece*. The network learned it because the composer built it in.

The theory in the weights is a theory of structural intent — not the composer's subjective intent (which is inaccessible), but the *structural intent* embodied in the harmonic relationships the composition actually contains. When Godspeed You! Black Emperor layers strings into a slowly building crescendo, the predictability of each added voice is what makes the accumulation feel inevitable rather than random. *COLUMNS* operates in a similar register: the frequency stacks are placed so that their overtone relationships form a redundant, self-reinforcing architecture. The network discovered this redundancy and encoded it as 17 weighted connections and 93 spectral modes.

### 8.4 Emergent Structure

Part 2's most striking finding was that instability is **emergent** — every individual component healthy ($\rho < 1$), but the coupled system exhibiting $\rho > 1$. Part 3 exhibits its own emergence: the harmonic relationships discovered by the network are not properties of any individual spectral frame — they are statistical regularities across the entire piece, visible only in the aggregate weight structure.

Both are invisible to decomposition-based analysis. Per-service monitoring misses emergent instability. Per-frame spectral analysis misses piece-spanning harmonic architecture. The trained model discovers emergent structure because it sees the whole — the whole system, the whole piece. The theory is holistic because the learning is holistic.

---

## 9. Limitations and Future Work

**Single-piece training.** The harmonic theory is specific to *COLUMNS*. This specificity is the point — the weight matrix is a fingerprint of *this* composition — but it raises the question of which features would persist across a corpus.

**Frame-by-frame independence.** No temporal memory. A recurrent architecture could exploit cross-frame context (e.g., persistent columns), potentially improving reconstruction.

**Linear approximation.** $W_{\text{eff}}$ linearizes a nonlinear network. The average-activation approach is partial; per-operating-point Jacobians (as in Part 2) would be more rigorous, trading generality for precision.

**Fixed tuning grid.** Just intonation, adaptive tuning, and microtonal systems define different measurement bases. Exploring non-equal-tempered collapse would reveal different harmonic structures in the same underlying music.

**Future directions:** Cross-piece comparative analysis (which structures are universal vs. piece-specific); learning the grid itself (discovering the observation angle that minimizes information loss for a given piece); perceptual validation (testing whether the network's theory preserves what is perceptually salient, not just spectrally measurable); integration with Part 2's temporal dynamics (applying Jacobian stability analysis to the temporal evolution of harmony).

---

## 10. The Theory in the Weights — Heard

This series began with a claim: trained model weights encode a theory of the data-generating process. Part 1 grounded the claim mathematically. Part 2 demonstrated it across four software systems. This essay has carried it into a domain where the encoded theory has a perceptual dimension — and where the relationship between theory and reality becomes audible.

The neural network trained on *COLUMNS* by Matthew Finch learned a theory of the piece's harmonic architecture:

- **17 harmonic relationships** at integer frequency ratios — overtone structure the composer wove into the spectral columns
- A **93×93 effective weight matrix** whose eigendecomposition reveals 93 resonant modes, all expansive, with a dominant mode at $\lvert\lambda\rvert = 18,807.87$
- **19.4% brightness recovery** — the portion predictable from the composer's harmonic language
- A **276.2-second audiation** — theory made audible

The collapse from 12-TET to 5-TET is a projection: measurement in a coarser basis destroying off-grid information. The expansion is a learned inverse: Bayesian posterior approximation given the collapsed observation and the training-data prior. The success of the expansion measures the *redundancy* in the original — the predictable structure the composer's choices imposed.

The V1/V2 comparison reveals that this theory can be expressed in fundamentally different internal geometries — distributed across 93 complex modes or concentrated into a single correction on an identity pathway — while achieving comparable reconstruction. The architecture is a language. The loss function is a hypothesis. The linearization carries assumptions. And through all of this, the numbers remain rigorously anchored to the actual spectral values of the music: MSE 0.00271 on training data, physics-informed corrections ensuring perceptual coherence, brightness recovery quantified to the tenth of a percent.

The irreversibility is real. Quantization's many-to-one nature prohibits perfect reconstruction. What the network recovers is not the original — it is the original's *theory*, as learned from data, through the particular angle of observation the instrument was built to occupy.

But the 19.4% recovered, and the 17 harmonic relationships encoding *how* — that is the composer's structural fingerprint, discovered by a neural network and written into its weights. The eigenvalues are the timbre. The weight matrix is the harmony. The reconstruction is the theory in sound.

[Listen: complete audiation on Bandcamp →](https://matthewrfinch.bandcamp.com/album/applied-weights-columns)

---

## Appendix A: Cross-Series Summary

For readers following the full series, this table maps the shared mathematical framework across all three domains.

| Component | Part 1 (General) | Part 2 (Software) | Part 3 (Audio) |
|-----------|------------------|-------------------|----------------|
| **Data** | Abstract $x_t$ | Microservice telemetry | Spectral frames |
| **Model** | Neural network | Mode-conditioned SSM | SpectralExpansionNet |
| **Theory in weights** | Compression → structure | Transition function | Harmonic map |
| **Key diagnostic** | SVD / eigendecomp. | Jacobian $\partial f/\partial z$ | $W_{\text{eff}}$ |
| **Spectral radius** | — | Stability ($\rho < 1$) | Amplification ($\rho \gg 1$) |
| **Eigenvalues** | Koopman modes | Growth/decay/oscillation | Resonant modes |
| **Eigenvectors** | Koopman eigenfunctions | Stability directions | Spectral directions |
| **Sparse features** | SAE decomposition | Causal/symptomatic/latent | Harmonic integer ratios |
| **Perceptual output** | Tables/plots | Stability maps, feature cards | Audiation WAV |

---

## References

- Bennett, W. R. (1948). *Spectra of Quantized Signals.* Bell System Technical Journal, 27(3), 446–472.
- Bricken, T., et al. (2023). *Towards Monosemanticity.* Anthropic.
- Brunton, S. L., et al. (2022). *Modern Koopman Theory for Dynamical Systems.* SIAM Review, 64(2), 229–340.
- Gray, R. M. & Neuhoff, D. L. (1998). *Quantization.* IEEE Trans. Information Theory, 44(6), 2325–2383.
- Griffiths, D. J. (2004). *Introduction to Quantum Mechanics.* 2nd ed. Prentice Hall.
- Koopman, B. O. (1931). *Hamiltonian Systems and Transformation in Hilbert Space.* PNAS, 17(5), 315–318.
- Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). *Deep learning for universal linear embeddings of nonlinear dynamics.* Nature Communications, 9, 4950.
- Oliver, B. M., Pierce, J. R., & Shannon, C. E. (1948). *The Philosophy of PCM.* Proc. IRE, 36(11), 1324–1331.
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal, 27(3), 379–423.
- Shannon, C. E. (1959). *Coding Theorems for a Discrete Source with a Fidelity Criterion.* IRE National Convention Record, 7(4), 142–163.
- Shwartz-Ziv, R. & Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information.* arXiv:1703.00810.
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). *The Information Bottleneck Method.* Proc. 37th Allerton Conference.
- Wootters, W. K. & Zurek, W. H. (1982). *A single quantum cannot be cloned.* Nature, 299, 802–803.
- Zurek, W. H. (2003). *Decoherence, einselection, and the quantum origins of the classical.* Rev. Modern Physics, 75(3), 715–775.
