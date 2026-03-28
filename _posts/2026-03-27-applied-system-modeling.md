---
layout: post
title: "Applied Theory: Reading System Dynamics from Learned Weights"
date: 2026-03-27
---

# Applied Theory: Reading System Dynamics from Learned Weights

*Part 2 of a three-part series on mechanistic modeling and learned structure*

> **Preceding**: [Weights as Theory: What Trained Models Actually Learn](/2026/03/24/introduction.html) — the general claim and its mathematical grounding
>
> **Following**: Part 3 — Hearing Structure: Compositional Analysis via Audio Synthesis

---

## 0. From Abstract Claim to Concrete Instrument

Part 1 of this series argued that trained model weights encode a *theory* of the data-generating process — not a theory in the symbolic, human-legible sense, but a compressed, distributed, operational one. The argument drew on compression theory (MDL, Kolmogorov complexity), the information bottleneck, and Koopman operator theory to establish that a model which generalizes well has necessarily discovered structure in the system it was fitted to, and that structure is encoded in the parameters.

That argument was intentionally abstract. This essay makes it concrete.

We take the "weights as theory" claim and test it end-to-end on four bounded software systems — microservice ecosystems with realistic topologies, failure modes, and coupling dynamics. For each system, we fit a neural state-space model (SSM) to simulated telemetry, then treat the trained model as a *diagnostic instrument*: extracting stability properties, causal attributions, early-warning indicators, and intervention targets entirely from the model's internal structure. No hand-crafted rules, no threshold tuning on raw metrics, no domain-specific feature engineering. The model learns the theory; we read it.

The pipeline produces, for each system, a structured analysis:

1. **Stability map** — Jacobian spectral radii across all operating points, quantifying where the system contracts perturbations and where it amplifies them
2. **Causal decomposition** — sparse autoencoder features classified as causal drivers, downstream symptoms, latent risks, or stabilizing mechanisms
3. **Intervention ranking** — features scored by ablation rescue, controllability, and temporal lead, yielding a prioritized shortlist of actionable targets
4. **Component coupling analysis** — per-subsystem stability tiers revealing that instability is emergent, not localized

Every one of these readouts is extracted from the model's *weights and derivatives*, not from the raw telemetry. The model is the instrument; the parameters are the theory. By the end, the claim from Part 1 is no longer an argument — it is an engineering methodology.

---

## 1. Four Systems Under Study

To test whether the pipeline generalizes beyond a single demo, we run it against four microservice ecosystems of increasing complexity, each representing a different operational domain.

### 1.1 Social Media Platform

The smallest system: 3 services (feed_service, notification_service, user_service), 1 database (posts_db), yielding 14 observables and 6 operating modes. Modes include `normal`, `content_spam_flood`, `notification_storm`, `db_replication_lag`, `viral_traffic_spike`, and `user_service_crash`. This is a compact, tightly-coupled system where cascading failures propagate quickly through a shallow call graph.

- **Dataset**: 1,200 episodes × 128 timesteps × 14 observables
- **Model**: latent_dim=12, hidden_dim=48, feature_dim=56

### 1.2 E-commerce Checkout

A mid-scale transactional system: 5 services (web_frontend, cart_service, inventory_service, payment_processor, shipping_estimator), 2 databases (products_db, transactions_db), producing 22 observables across 7 modes. Failure scenarios include `payment_gateway_timeout`, `inventory_desync`, `flash_sale_surge`, `fraud_detection_overload`, `shipping_api_failure`, and `db_connection_exhaustion`. The system has a clear pipeline topology — request flows from frontend through cart, payment, and shipping — making it a natural test for whether the model discovers directional coupling.

- **Dataset**: 1,500 episodes × 128 timesteps × 22 observables
- **Model**: latent_dim=16, hidden_dim=64, feature_dim=80

### 1.3 IoT Telemetry Pipeline

A data-intensive ingestion system: 6 services (ingestion_gateway, stream_processor, device_registry, alerting_engine, analytics_aggregator, api_server), 2 databases (timeseries_db, device_metadata_db), with 27 observables and 8 modes. Failure modes include `ingestion_flood`, `stream_backpressure`, `device_registry_corruption`, `alert_storm`, `timeseries_write_saturation`, `cold_start_lag`, and `network_partition`. The distinguishing feature is high fan-in at the ingestion layer and high fan-out at the alerting layer, creating asymmetric propagation dynamics.

- **Dataset**: 1,500 episodes × 128 timesteps × 27 observables
- **Model**: latent_dim=18, hidden_dim=72, feature_dim=88

### 1.4 Financial Order Processing

The most complex system: 7 services (gateway, order_router, matching_engine, risk_validator, settlement_service, compliance_logger, market_data_feed), 3 databases (orders_db, positions_db, audit_store), generating 33 observables across 9 modes. Failure modes span `matching_engine_halt`, `market_data_stale`, `risk_model_timeout`, `settlement_backlog`, `network_partition`, `orders_db_degraded`, `flash_crash_cascade`, and `compliance_deadlock`. This system has the deepest call chains, the most databases, and the richest failure taxonomy — a stress test for the pipeline's scalability.

- **Dataset**: 1,800 episodes × 128 timesteps × 33 observables
- **Model**: latent_dim=20, hidden_dim=80, feature_dim=100

### Why Four Systems?

A single demonstration proves the pipeline *works*. Four demonstrations — spanning 3 to 7 services, 6 to 9 modes, 14 to 33 observables — test whether the pipeline *generalizes*. If the structural patterns we extract hold across systems of different topologies, scales, and failure taxonomies,  the claim that "trained weights encode usable theory" is not anecdotal but structural: the same compression discovers the same kinds of architecture regardless of system topology.

---

## 2. The Model: A Mode-Conditioned Neural State-Space Model

### 2.1 State-Space Formulation

The core model is a neural state-space model (SSM) that learns a latent dynamical system from observed telemetry:

$$z_{t+1} = f_\theta(z_t,\, x_t,\, \phi(m_t)), \qquad \hat{x}_t = g_\theta(z_t,\, \phi(m_t))$$

where $z_t \in \mathbb{R}^d$ is a learned latent state, $x_t \in \mathbb{R}^p$ is the observation vector, $m_t$ is the operating mode, $\phi(m_t)$ is a learned mode embedding, and $\theta$ denotes all trainable parameters. The model simultaneously learns two things:

1. **A coordinate system** — the mapping from high-dimensional telemetry $x_t$ into a compact latent state $z_t$, where the system's dynamics simplify
2. **A transition law** — the function $f_\theta$ that describes how the system evolves forward in time, conditioned on the current operating mode

The connection to Koopman operator theory from Part 1 is direct. For a dynamical system $x_{t+1} = F(x_t)$, the Koopman operator $\mathcal{K}$ acts on observable functions $g$ by composition: $(\mathcal{K}g)(x) = g(F(x))$. The operator is linear even when $F$ is nonlinear — it trades state-space nonlinearity for infinite-dimensional linearity in the space of observables. Koopman eigenfunctions $\phi_k$ satisfy $\phi_k(x_{t+1}) = \lambda_k \phi_k(x_t)$: each evolves by scalar multiplication, providing coordinates in which the dynamics are linear [Koopman, 1931; Brunton et al., 2022].

A learned latent space where $z_{t+1} \approx A_m z_t$ under mode $m$ is a finite-dimensional Koopman approximation: the encoder discovers candidate eigenfunctions, the transition function approximates the Koopman operator, and the eigenvalues of the linearized transition encode growth rates, decay rates, and oscillation frequencies. Deep Koopman methods [Lusch et al., 2018] and recent work connecting SSMs to Koopman embeddings [Hašić & Edalat, 2024] confirm that this is not merely an analogy — neural dynamical models implicitly learn Koopman-like structure when trained on sufficient data.

### 2.2 FiLM Conditioning

The mode influences the transition function through Feature-wise Linear Modulation (FiLM) [Perez et al., 2018]. Rather than concatenating the mode embedding with the input — which mixes mode information with state information in an entangled way — FiLM applies a learned affine transformation to the hidden representation:

$$h = f_{\text{base}}([z_t \| x_t]), \qquad z_{t+1} = f_{\text{out}}\bigl((\text{scale}_m + 1) \odot h + \text{shift}_m\bigr)$$

where $\text{scale}_m$ and $\text{shift}_m$ are mode-specific parameters produced by a small network applied to the mode embedding. The $+1$ initialization on scale ensures the model begins mode-agnostic — all modes share the same dynamics at initialization — and learns mode-specific adjustments. The mode acts as a *coordinate warp* on the dynamics: a rotation and translation of the hidden representation that steers the transition toward mode-specific behavior while preserving a shared latent geometry.

This architectural choice is deliberate and theory-motivated. In Koopman terms, different operating modes correspond to different dynamical regimes with different spectral decompositions. FiLM conditioning allows the model to learn a *family* of transition operators indexed by mode, sharing a common set of Koopman coordinates but with mode-dependent eigenvalues — exactly the structure we need for stability analysis across regimes.

### 2.3 Direct Prediction (No Residual Skip)

A common design for SSMs uses a residual connection: $z_{t+1} = z_t + \Delta f_\theta(z_t, x_t, m_t)$. This is easier to train — the identity provides a gradient highway, and the network only needs to learn the *correction* — but it cripples downstream stability analysis. With a residual connection, the Jacobian of the transition is:

$$J_{\text{residual}} = I + \frac{\partial \Delta f}{\partial z}$$

The identity matrix dominates: eigenvalues cluster near 1, and the spectral radius is uninformative regardless of the actual dynamics. With direct prediction, the Jacobian is the *complete* learned dynamics:

$$J_{\text{direct}} = \frac{\partial f_\theta}{\partial z}$$

This is a deliberate trade — harder to train, but the weights encode a full theory of the transition function, not a perturbative correction atop an assumed-stable baseline. Every eigenvalue of $J_{\text{direct}}$ is learned, making spectral analysis meaningful.

### 2.4 Mode Embedding Dropout

If the FiLM conditioning always receives the true mode, a training shortcut emerges: the latent state $z_t$ does not need to encode mode-discriminating information from the dynamics, because the mode embedding does all the work. The contrastive loss becomes trivially satisfied (same FiLM parameters → similar hidden states), and the mode classification head learns to read the conditioning rather than the latent geometry.

We break this shortcut with mode embedding dropout. During training, with probability $p_{\text{drop}} = 0.3$ per TBPTT chunk, the mode embedding is zeroed in the transition function while preserved in the observation decoder:

$$z_{t+1} = f_\theta(z_t, x_t, \mathbf{0}), \qquad \hat{x}_t = g_\theta(z_t, \phi(m_t))$$

This forces the latent state to carry mode information derived from the *dynamics alone*. The observation decoder can still use the true mode for reconstruction, but the transition — and therefore the Jacobian — must learn to distinguish modes from the temporal evolution of the state. Empirically, mode dropout was the key fix that resolved degenerate training: without it, mode classification loss collapsed to zero immediately (the head read the conditioning) and contrastive loss provided no gradient (all hidden states were similar under identical FiLM parameters).

### 2.5 Training Objective

The model is trained with a composite loss that serves multiple purposes simultaneously:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{prediction}} + \lambda_1 \underbrace{\mathcal{L}_{\text{CE}}}_{\text{mode structure}} + \lambda_2 \underbrace{\mathcal{L}_{\text{NT-Xent}}}_{\text{geometry}} + \lambda_3 \underbrace{\mathcal{R}_{\text{jac}}}_{\text{stability}}$$

**Reconstruction loss** ($\mathcal{L}_{\text{recon}}$): MSE between predicted and actual observations, averaged over TBPTT chunks of 24 timesteps. This is the primary signal: the model must predict the system's telemetry.

**Mode classification loss** ($\mathcal{L}_{\text{CE}}$, $\lambda_1 = 0.35$): cross-entropy from a linear head applied to the latent state $z_{t+1}$. Forces mode-discriminating information into the latent representation — the model cannot achieve low mode CE unless $z_t$ encodes *which regime the system is in*.

**Contrastive loss** ($\mathcal{L}_{\text{NT-Xent}}$, $\lambda_2 = 0.1$): the NT-Xent loss [Chen et al., 2020] applied to the final latent state of each TBPTT chunk, with mode identity as the positive-pair criterion. This pulls same-mode trajectory endpoints together and pushes different-mode endpoints apart, inducing a geometric structure in latent space where mode clusters are separated — a prerequisite for meaningful mode-conditioned analysis.

The NT-Xent loss operates on $L_2$-normalized latent vectors with a temperature parameter $\tau = 0.1$:

$$\mathcal{L}_{\text{NT-Xent}} = -\frac{1}{|P|} \sum_{(i,j) \in P} \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $P$ is the set of positive pairs (same mode) and $\text{sim}(a,b) = a^\top b / \|a\|\|b\|$.

**Jacobian spectral penalty** ($\mathcal{R}_{\text{jac}}$, $\lambda_3 = 0.5$): a regularizer on the spectral norm of the linearized normal-mode transition. The penalty targets the operator 2-norm of the composite weight matrix $W_{\text{normal}} = W_3 \cdot \text{diag}(\text{scale}_{\text{normal}}) \cdot W_2 \cdot W_1$, penalizing it when it exceeds a target $\rho_{\text{target}} = 0.80$:

$$\mathcal{R}_{\text{jac}} = \max\bigl(0,\, \|W_{\text{normal}}\|_2 - \rho_{\text{target}}\bigr)^2$$

This prevents the normal operating mode from dominating the instability landscape — without it, the normal mode's spectral radius drifts upward, making it difficult to distinguish healthy operation from genuine failure modes in downstream analysis.

Training uses TBPTT with chunk size 24, Adam optimizer (learning rate $3 \times 10^{-4}$), and runs for 750 epochs for the SSM. The combination of these four losses shapes the latent space into a representation that is simultaneously *predictive* (low reconstruction error), *mode-structured* (discriminable regimes), *geometrically organized* (contrastive clustering), and *stability-calibrated* (normal mode contractive). Each property is essential for the downstream analysis.

---

## 3. Sparse Autoencoder: Decomposing the Learned Representation

The SSM gives us dynamics — a latent space where the system's evolution is encoded. But the individual latent dimensions are not interpretable: they are an arbitrary basis for the learned coordinate system, entangled by the training process. To build a *vocabulary* for describing what the dynamics are doing at each moment, we apply a sparse autoencoder (SAE) to the latent representations.

The SAE learns an overcomplete dictionary of features — directions in latent space that activate sparsely across timesteps:

$$h = \text{ReLU}(W_{\text{enc}} \cdot z + b_{\text{enc}}), \qquad \hat{z} = W_{\text{dec}} \cdot h + b_{\text{dec}}$$

$$\mathcal{L}_{\text{SAE}} = \|z - \hat{z}\|^2 + \lambda_s \|h\|_1$$

The reconstruction term ensures the dictionary spans the latent space; the $L_1$ penalty on activations $h$ ensures sparsity — at any given timestep, only a small subset of features is active. The decoder columns $W_{\text{dec}}$ are renormalized to unit norm after each gradient step, ensuring features represent *directions* in latent space rather than arbitrary scalings [Bricken et al., 2023].

In our experiments, the SAE is overcomplete by a factor of roughly 4–5× (e.g., 80 features for a 16-dimensional latent space), trained for 500 epochs with sparsity penalty $\lambda_s = 0.02$. Across the four systems, convergence yielded active feature rates of 15–20%, meaning roughly 12–16 of 80 features are active at any given timestep — sparse enough to be interpretable, rich enough to cover the observable dynamics.

Each feature is auto-labeled by its correlation with system modes and observables. A feature labeled `flash_sale_surge_cart_service.queue_depth+|web_frontend.session_count+` activates when the system is in flash-sale mode *and* when cart queue depth and session counts are elevated. These labels are not imposed — they are discovered from the statistical relationship between feature activations and the raw telemetry.

The connection to Part 1 is direct: SAE features are candidate Koopman eigenfunctions — observable combinations that evolve coherently under the learned dynamics. Where Koopman theory decomposes the system into modes of temporal evolution, the SAE decomposes the latent representation into modes of *spatial structure* — directions in which the system's state varies meaningfully. The two decompositions are complementary, and their intersection (features aligned with dynamical eigenvectors) is where the most mechanistically informative diagnostics emerge.

---

## 4. Reading the Theory: Stability Analysis

This is where "weights as theory" ceases to be a philosophical position and becomes an engineering tool. The Jacobian of the learned transition function — a derivative of the weight structure, computable via autograd — encodes the system's local stability properties at every operating point.

### 4.1 The Jacobian and Spectral Radius

At any point along a trajectory, autograd computes the Jacobian of the learned transition:

$$A_t = \frac{\partial f_\theta}{\partial z}\bigg|_{(z_t, x_t, \phi(m_t))}$$

The spectral radius $\rho(A_t) = \max_i |\lambda_i(A_t)|$ — the magnitude of the largest eigenvalue — is the local stability score:

- $\rho < 1$: perturbations contract. The system *locally* tends to return toward its trajectory.
- $\rho \approx 1$: marginally stable. Small perturbations neither grow nor decay.
- $\rho > 1$: perturbations grow exponentially. The system is locally unstable.

This is not a metric about the model's training accuracy. It is a *measurement extracted from the model's theory* about the real system's dynamics. The learned transition function $f_\theta$ encodes, through its parameterization, how the system responds to infinitesimal perturbations at each operating point — and the eigenstructure of the Jacobian makes that response quantitative.

### 4.2 Stability Tiers

Rather than applying fixed thresholds, we calibrate stability tiers from the empirical distribution of spectral radii within each system — defining HEALTHY, STRESSED, and CRITICAL regions based on the 33rd and 67th percentiles. This respects the fact that different systems operate at different baseline contraction rates.

### 4.3 Cross-System Stability Results

The stability maps across all four systems reveal a consistent structural pattern:

| System | Mean $\rho$ | Std | Max $\rho$ | Lyap. mean | Lyap. max |
|--------|-------------|-----|------------|------------|-----------|
| Social Media | 0.467 | 0.217 | 1.610 | −0.857 | +0.237 |
| E-commerce | 0.457 | 0.245 | 1.579 | −0.888 | +0.327 |
| IoT Telemetry | 0.415 | 0.201 | 1.859 | −0.969 | +0.273 |
| Financial Orders | 0.495 | 0.199 | 2.882 | −0.772 | +0.190 |

Several patterns are immediately evident:

**Contractive in expectation, locally unstable.** Every system is contractive on average (negative Lyapunov exponents), yet every system has operating points where $\rho > 1$ — the system is locally amplifying perturbations. This is the signature of *intermittent instability*: the system spends most of its time in stable regimes but passes through critical windows where failures can amplify.

**Complexity scales with instability.** The financial orders system — the most complex, with 7 services, 3 databases, and 9 modes — exhibits both the highest mean spectral radius (0.495) and the highest maximum (2.882). The IoT system, despite having more services than e-commerce, shows *lower* mean $\rho$ but *higher* peak $\rho$, suggesting that its instability is more concentrated in fewer, more severe episodes. This matches the domain: IoT systems under backpressure either handle load gracefully or saturate catastrophically.

**Tier distributions are consistent.** Across all four systems, the three stability tiers (strongly contractive, weakly contractive, near-critical) are distributed approximately 33/33/33, which is an artifact of the percentile-based tier calibration — but the *shapes* of the distributions within tiers differ meaningfully. The financial system has a heavier right tail (more extreme instability events), while the IoT system has the sharpest transition from contractive to critical.

### 4.4 Per-Mode Stability Signatures

The most diagnostic readout is the per-mode mean spectral radius — which failure types push the system closest to instability. Across all four systems:

**Social Media:**
| Mode | Mean $\rho$ | Interpretation |
|------|-------------|----------------|
| db_replication_lag | 0.317 | Most stable fault — slow drift, well-contained |
| normal | 0.406 | Healthy baseline |
| user_service_crash | 0.408 | Near-normal — crash is abrupt but localized |
| content_spam_flood | 0.512 | Moderate — load-driven, propagates through feed |
| viral_traffic_spike | 0.498 | Similar — scaling pressure |
| notification_storm | 0.674 | **Most destabilizing** — cascading amplification |

**E-commerce:**
| Mode | Mean $\rho$ | Interpretation |
|------|-------------|----------------|
| fraud_detection_overload | 0.335 | Most stable — isolated compute bottleneck |
| payment_gateway_timeout | 0.355 | Well-contained — timeouts limit propagation |
| normal | 0.395 | Baseline |
| inventory_desync | 0.476 | Moderate — state inconsistency propagates |
| shipping_api_failure | 0.514 | External dependency failure |
| db_connection_exhaustion | 0.526 | Resource starvation — broad impact |
| flash_sale_surge | 0.581 | **Most destabilizing** — all components loaded simultaneously |

**IoT Telemetry:**
| Mode | Mean $\rho$ | Interpretation |
|------|-------------|----------------|
| device_registry_corruption | 0.296 | Most stable — affects metadata, not data flow |
| normal | 0.338 | Baseline |
| ingestion_flood | 0.351 | Near-normal — gateway absorbs |
| timeseries_write_saturation | 0.394 | Moderate — storage bottleneck |
| alert_storm | 0.417 | Feedback amplification through alerting |
| stream_backpressure | 0.490 | Significant — blocks data pipeline |
| network_partition | 0.498 | Broad connectivity failure |
| cold_start_lag | 0.525 | **Most destabilizing** — cascading initialization delays |

**Financial Orders:**
| Mode | Mean $\rho$ | Interpretation |
|------|-------------|----------------|
| market_data_stale | 0.317 | Most stable — stale data is detectable, contained |
| orders_db_degraded | 0.381 | Database degradation well-handled |
| normal | 0.461 | Baseline |
| network_partition | 0.442 | Contained — partition protocols engage |
| matching_engine_halt | 0.528 | Significant — core processing stopped |
| settlement_backlog | 0.524 | Queue buildup with long tail (max $\rho$ = 2.882) |
| flash_crash_cascade | 0.538 | Market-wide amplification |
| compliance_deadlock | 0.556 | **Most destabilizing** — regulatory + operational coupling |
| risk_model_timeout | 0.698 | **Highest mean $\rho$** — blocks order flow, cascades to matching |

A structural finding emerges: **the most destabilizing failure modes are consistently those that involve cascading or system-wide coupling**, not those with the most dramatic local symptoms. In the financial system, `risk_model_timeout` has the highest mean $\rho$ (0.698) — not because risk validation is the most critical service in isolation, but because its timeout blocks order flow through the matching engine, settlement, and compliance, creating a multi-service amplification cascade. In e-commerce, `flash_sale_surge` is most destabilizing because it simultaneously loads *every* component — the instability is not in any single service but in the coordinated stress.

This is precisely the kind of insight that raw per-service monitoring cannot provide: the model's learned transition function encodes the *coupling structure* of the system, and the Jacobian eigenvalues quantify how that coupling amplifies or suppresses perturbations in each regime.

---

## 5. Causal Decomposition: Which Features Drive Instability?

The stability map tells us *that* the system becomes unstable and *in which modes*. The next question is *why* — which specific observable patterns are driving the instability? This is where the SAE features become essential. By analyzing how feature activations relate to spectral radius — through both correlation and geometric alignment — we can decompose the instability signal into mechanistic categories.

### 5.1 Feature-Stability Correlation

For each SAE feature $k$, we compute the conditional spectral radius difference:

$$\Delta\rho_k = \mathbb{E}[\rho \mid f_k > \tau] - \mathbb{E}[\rho \mid f_k \leq \tau]$$

Features with $\Delta\rho > 0.05$ are classified as destabilizing; $\Delta\rho < -0.05$ as stabilizing; otherwise neutral. The distribution across systems:

| System | Destabilizing | Neutral | Stabilizing | Feature Dim |
|--------|--------------|---------|-------------|-------------|
| Social Media | 46% | 29% | 25% | 56 |
| E-commerce | 44% | 24% | 32% | 80 |
| IoT Telemetry | 42% | 33% | 25% | 88 |
| Financial Orders | 53% | 22% | 25% | 100 |

The consistent pattern: roughly 40–55% of learned features are associated with increased instability. But correlation is not causation — a feature could be destabilizing either because it *drives* instability (it appears in the Jacobian's amplifying eigenvector) or because it *appears during* instability as a downstream symptom. Distinguishing these requires the next step.

### 5.2 Eigenvector Alignment: Geometry, Not Just Correlation

At critical operating points (the top decile of $\rho$), the dominant eigenvector $v_{\max}$ of the Jacobian $A_t$ points in the direction the system is *actually being amplified* — the direction of maximum perturbation growth. We align each SAE feature's decoder column $d_k$ with this eigenvector:

$$\text{align}_k = \langle |d_k \cdot v_{\max}| \rangle_{\text{critical}}$$

A feature aligned with $v_{\max}$ (alignment score $> 0.3$) lies along the instability direction in latent space. A feature with high $\Delta\rho$ but low alignment is correlated with instability but doesn't point in the direction of amplification — it is a *downstream marker*, not a driver.

### 5.3 The Five-Way Classification

Crossing the stability role ($\Delta\rho$) with eigenvector alignment produces five mechanistic categories:

| Category | $\Delta\rho$ | Aligned? | Interpretation |
|----------|-------------|----------|----------------|
| **causal** | destabilizing | yes | Actively driving instability — lies on the amplifying direction |
| **symptomatic** | destabilizing | no | Correlated with instability but downstream — a marker, not a driver |
| **latent_risk** | neutral | yes | Points along the instability direction but currently inactive — a dormant threat |
| **stabilizing_aligned** | stabilizing | yes | Actively suppresses the dominant unstable mode |
| **irrelevant** | neither | no | Not involved in stability dynamics |

The distribution across our four systems:

| Category | Social Media | E-commerce | IoT | Financial |
|----------|-------------|------------|-----|-----------|
| causal | 25% | 14% | 8% | 10% |
| symptomatic | 21% | 30% | 34% | 43% |
| latent_risk | 11% | 6% | 2% | 0% |
| stabilizing_aligned | 14% | 10% | 3% | 1% |
| irrelevant | 29% | 40% | 52% | 46% |

A striking scaling pattern: **as system complexity increases, the ratio of symptomatic to causal features grows**. In the compact social media system (3 services), causal features outnumber symptomatic ones. In the financial system (7 services), symptomatic features outnumber causal ones by more than 4-to-1. This is exactly what dynamical systems theory predicts: in larger, more coupled systems, each root cause triggers a wider cascade of downstream effects, inflating the symptomatic-feature count while the number of actual driving directions remains small.

This has an immediate practical implication: **in complex systems, most destabilizing signals in your telemetry are symptoms, not causes**. An alerting system that triggers on any destabilizing feature will generate false urgency on symptoms while potentially missing the actual drivers. The five-way classification cuts through this: it separates the causal few from the symptomatic many using the model's internal geometry.

### 5.4 Concrete Causal Findings

To illustrate the diagnostic value, here are selected causal features from each system:

**Social Media**: F29 ($\Delta\rho = +0.714$, top destabilizer) — labeled `user_service_crash_notification_service.delivery_rate+`. The model learned that rising notification delivery rate during a user service crash is not merely a symptom of load spillover — it is a *causal amplifier*: the notification service, attempting to deliver queued messages to a failing user service, creates a feedback loop that drives the system toward instability. The feature's decoder column aligns with the Jacobian's dominant eigenvector, confirming that this specific coupling direction is where perturbation growth occurs.

**E-commerce**: F38 ($\text{rescue} = +0.065$, top rescue score) — labeled `db_connection_exhaustion_products_db.replication_lag−|products_db.query_latency−|products_db.connection_pool_pct−`. This feature activates when products_db metrics are *decreasing* during connection exhaustion — counterintuitive, since we might expect database metrics to spike. The model discovered that the *drop* in database activity (connections exhausted → queries can't execute → metrics collapse) is the causal mechanism, not the spike in error rates that follows.

**IoT**: F24 ($\Delta\rho = +0.461$, top destabilizer) — labeled `cold_start_lag_analytics_aggregator.error_rate+|api_server.throughput−|analytics_aggregator.latency+`. Cold start creates cascading initialization delays where the analytics aggregator fails and the API server's throughput drops. The model identified this compound pattern — not any single metric — as the causal signature.

**Financial**: F62 ($\Delta\rho = +0.513$) — labeled `flash_crash_cascade_matching_engine.order_fill_rate+|market_data_feed.latency−|matching_engine.latency−`. During a flash crash, the model found that the *increase* in fill rate combined with *decreased* latency — the matching engine speeding up during volatile conditions — is part of the amplification mechanism. This is a learned feature that encodes a *compound behavioral pattern*, not a single metric.

### 5.5 Ablation as Causal Confirmation

Correlation and alignment identify *candidate* causal features. Ablation provides a causal *test*. For each feature $k$, we zero its SAE activation at critical points, reconstruct the modified latent state through the decoder, recompute the Jacobian, and measure the change in spectral radius:

$$\text{rescue}(k) = \mathbb{E}_{t:\,\rho_t \geq \rho_{90\%}}\bigl[\rho(A_t) - \rho(A_t \mid f_k = 0)\bigr]$$

A positive rescue score means *removing this feature direction from the latent state reduces instability* — the feature is load-bearing in the amplification mechanism. This is a causal intervention on the model's internal representation, the mechanistic interpretability equivalent of a controlled experiment.

Top rescue scores across systems:

| System | Feature | Rescue | $\Delta\rho$ | Label |
|--------|---------|--------|-------------|-------|
| Social Media | F14 | +0.073 | destabilizing | — |
| E-commerce | F38 | +0.065 | +0.521 | db_connection_exhaustion (products_db) |
| IoT | F53 | +0.058 | +0.054 | causal (unlabeled) |
| Financial | F91 | +0.033 | destabilizing | matching_engine_halt |

The ablation also extends to *early intervention*: zeroing features $\tau$ steps *before* the critical point and rolling the SSM forward to check whether the crisis is averted. In the IoT system, F79 (`cold_start_lag_api_server.latency+|api_server.throughput−|api_server.error_rate+`) achieves an early-rescue rate of 17.2% at $\tau \in \{2, 3, 5\}$ steps — meaning that in roughly 1 in 6 approaching crises, intervening on this single feature direction would prevent the instability from developing.

### 5.6 Early-Warning via Lead-Lag Analysis

Cross-correlation between feature activations and spectral radius at multiple time lags identifies leading indicators — features that activate *before* instability rises:

$$r_k(\tau) = \text{Corr}\bigl(f_k(t),\, \rho(t + \tau)\bigr), \qquad \tau \in [-10, +10]$$

Features whose peak correlation occurs at positive lag are *leading*: they activate before $\rho$ rises, providing a prediction window.

| System | Leading | Lagging | Simultaneous | Uncorrelated | Max Lead |
|--------|---------|---------|-------------|--------------|----------|
| Social Media | 9 | 26 | 2 | 19 | +8 steps |
| E-commerce | 14 | 36 | 4 | 26 | +8 steps |
| IoT | 16 | 25 | 3 | 44 | +6 steps |
| Financial | 9 | 45 | 4 | 42 | +10 steps |

The financial system yields the longest lead time (+10 steps for F44, a symptomatic destabilizer, the top-ranked feature by composite importance), consistent with its deeper call chains: perturbations take longer to propagate through 7 services, creating a wider prediction window. The IoT system has the *most* leading features (16 of 88) but shorter lead times, reflecting its fan-out topology where perturbations reach many components quickly but shallowly.

---

## 6. Controllability: What Can You Actually Act On?

Knowing what *causes* instability is necessary but not sufficient for operational response. Controllability asks: which causal directions can the system's inputs actually *reach*? A feature might identify the root cause perfectly, but if that latent direction is not reachable from the available control inputs, the insight is diagnostic but not actionable.

### 6.1 The Controllability Gramian

For each mode, we compute the discrete-time controllability Gramian:

$$\mathcal{C}_m = \sum_{i=0}^{n-1} A_m^i B_m (A_m^i B_m)^\top$$

where $A_m = \langle \partial f / \partial z \rangle_m$ and $B_m = \langle \partial f / \partial x \rangle_m$ are mode-averaged Jacobians over horizon $n$ (set to the latent dimension). The eigenvalues of $\mathcal{C}_m$ describe how many independent latent directions are reachable from observation-space inputs. Directions with eigenvalues below 1% of the maximum are considered uncontrollable.

### 6.2 Controllable Dimensionality

| System | Mode | Controllable Dims | Top-3 Share |
|--------|------|-------------------|-------------|
| Social Media | user_service_crash | 6 | 91.6% |
| Social Media | notification_storm | 8 | — |
| E-commerce | fraud_detection_overload | 7 | 84.4% |
| E-commerce | shipping_api_failure | 8 | 88.7% |
| IoT | cold_start_lag | 9 | 76.1% |
| IoT | stream_backpressure | 12 | 66.3% |
| Financial | normal | 12 | 66.5% |
| Financial | network_partition | 11 | 76.4% |

Two patterns stand out:

**Concentrated vs. distributed controllability.** In the social media system, `user_service_crash` has only 6 controllable dimensions with a top-3 eigenvalue share of 91.6% — nearly all controllability is concentrated in three directions. This means precise, targeted interventions are possible but the intervention space is narrow. In contrast, `stream_backpressure` in the IoT system has 12 controllable dimensions with top-3 share of only 66.3% — controllability is distributed across many directions, suggesting that broad, multi-signal interventions are needed rather than single-point fixes.

**Mode-dependent controllability structure.** The same system has different controllability profiles in different modes — an intervention that works in one failure mode may not work in another. The financial system is notable: `normal` mode has 12 controllable dimensions (full controllability), but some fault modes drop to 10-11, indicating that certain failure types *restrict* the system's controllability, making it harder to steer back to health.

### 6.3 The Actionability Funnel

Combining causal classification, controllability scoring, and lead-lag timing produces a four-tier actionability classification:

| Category | Criteria | Interpretation |
|----------|----------|----------------|
| **fully_actionable** | causal + controllable + leading ($\text{lag} \geq 2$) | The intervention shortlist: you can detect it early, it matters, and you can push against it |
| **actionable_no_lead** | causal + controllable but lagging/simultaneous | Reactive but effective: real-time intervention is possible |
| **detectable_uncontrollable** | causal but not controllable | Diagnostic value: explains root cause but requires architectural change, not operational response |
| **not_actionable** | insufficient rescue, controllability, or both | Background noise for this analysis |

Distribution across systems:

| System | Fully Actionable | Actionable (no lead) | Not Actionable |
|--------|-----------------|---------------------|----------------|
| Social Media | 9% | 61% | 30% |
| E-commerce | 4% | 55% | 41% |
| IoT | 3% | 42% | 55% |
| Financial | 4% | 46% | 50% |

The fully actionable features are rare — 3–9% of all features — but they are the highest-value output of the entire pipeline. These are features where all three conditions align: the model identifies them as causal drivers (not symptoms), they point along latent directions reachable from system inputs (not architectural constraints), and they activate *before* the instability develops (providing intervention lead time).

The inverse scaling of fully-actionable features with system complexity is not a failure of the method — it reflects reality. In complex systems, most instability directions are either not controllable from available inputs (they require architectural redesign) or do not provide advance warning (the coupling is too tight for temporal separation). The pipeline's value is in *finding* the rare actionable features that exist, and in clearly categorizing the non-actionable ones so engineers know *why* they cannot be directly addressed.

---

## 7. Component Coupling and Emergent Instability

### 7.1 Per-Component Stability

A natural question: which *component* is unstable? The pipeline answers this by isolating each subsystem's contribution to the Jacobian and computing component-local spectral radii. The result is unambiguous:

**Social Media**: feed_service $\rho = 0.145$, notification_service $\rho = 0.222$, user_service $\rho = 0.270$, posts_db $\rho = 0.052$ — **all HEALTHY**.

**E-commerce**: payment_processor $\rho = 0.282$ (highest), products_db $\rho = 0.048$ (lowest) — **all HEALTHY**.

**IoT**: device_registry $\rho = 0.195$ (highest), device_metadata_db $\rho = 0.055$ (lowest) — **all HEALTHY**.

**Financial**: risk_validator $\rho = 0.212$ (highest), compliance_logger $\rho = 0.065$ (lowest) — **all HEALTHY**.

Every component, in every system, is individually healthy. Yet all four systems exhibit operating points where the global spectral radius exceeds 1.0, sometimes substantially ($\rho_{\max} = 2.88$ in the financial system). **Instability is emergent** — it arises from the coupling between components, not from any single component's local dynamics.

This is the most important structural finding of the analysis, and it is invisible to standard monitoring approaches that track per-service health independently. The model's learned transition function encodes the *complete coupled dynamics*, including off-diagonal Jacobian blocks that describe how perturbations in one component amplify through others. The instability lives in the *topology*, not in the *nodes*.

### 7.2 Soft Mode Assignment

The SSM assigns a soft probability distribution over modes at each timestep, based on prediction-error affinity — which mode-specific transition function best explains the observed dynamics. The entropy of this assignment, $H_t = -\sum_m p(m|z_t) \log p(m|z_t)$, provides a diagnostic in its own right:

| System | Mean Entropy | P95 Entropy | Max Entropy |
|--------|-------------|-------------|-------------|
| Social Media | 0.545 | — | 1.391 |
| E-commerce | 0.442 | 0.929 | 1.097 |
| IoT | 0.501 | 1.045 | 1.352 |
| Financial | 0.296 | 0.819 | 1.175 |

Low entropy means the model is confident about the operating regime; high entropy indicates a transition zone — the system is between modes, and the dynamics are ambiguous. The financial system has the *lowest* mean entropy (0.296), suggesting its 9 modes are the most sharply distinguished by the learned dynamics. The IoT system has the highest P95 entropy (1.045), indicating more frequent ambiguous transitions — consistent with a data pipeline where modes blend (e.g., backpressure can coexist with write saturation).

Mode share distributions are approximately uniform across all systems (10–16% per mode), confirming that mode dropout and contrastive training successfully prevented mode collapse during training.

---

## 8. The Feature Card: Unified Diagnostic Output

The pipeline's headline output is the ranked feature card — a composite importance score that integrates all analysis layers:

$$I_k = 0.35 \cdot \frac{|\text{rescue}_k|}{\max|\text{rescue}|} + 0.20 \cdot \text{align}_k + 0.15 \cdot \frac{|\Delta\rho_k|}{\max|\Delta\rho|} + 0.15 \cdot \text{ctrl}_k + 0.15 \cdot \text{lead\_bonus}_k$$

where $\text{lead\_bonus}_k = \text{peak\_lag}_k / \max(\text{lag})$ for leading features and 0 otherwise.

### Top-Ranked Features Across Systems

**Social Media** — top feature: F29 (importance 0.616), a *causal destabilizer* representing the notification-delivery-during-crash feedback loop. The top *latent risk* feature is F48 (importance 0.576) — currently neutral in $\Delta\rho$ but aligned with the instability eigenvector, meaning it identifies a dormant failure direction not captured by current monitoring.

**E-commerce** — top feature: F38 (importance 0.695), the `db_connection_exhaustion` causal driver described earlier. The second feature, F30 (importance 0.606), is labeled `multi_payment_processor.error_rate−` — a mode-spanning pattern (the `multi_` prefix) where *decreased* payment error rate is paradoxically causal. The model learned that this pattern (errors dropping because connections are exhausted and no requests get through) is a load-bearing part of the instability mechanism.

**IoT** — top feature: F55 (importance 0.590), an unlabeled symptomatic destabilizer with the longest lead time (+6 steps). Unlabeled features are not failures of the pipeline — they represent *compound patterns* in latent space that don't map cleanly to individual observables. F55 encodes a combination of behaviors across multiple services that collectively precede instability. The second feature, F70 (importance 0.566), is labeled `cold_start_lag_analytics_aggregator.error_rate+|analytics_aggregator.latency+|api_server.error_rate+` — the analytics-layer cascade during cold start.

**Financial** — top feature: F44 (importance 0.559), an unlabeled symptomatic destabilizer with a 10-step lead time — the earliest warning signal across all four systems. Feature F49 (importance 0.489), a *causal* feature, has the highest rescue score (+0.025) and high controllability (0.841) — making it the most mechanistically actionable feature in the financial system despite ranking third overall.

### Reading the Cards: A Decision Framework

The feature cards directly map to engineering decisions:

1. **Fully actionable + leading** → configure automated monitoring triggers. These features provide advance warning of instability through directions the system can respond to. Set thresholds on the SAE feature activation, not on raw metrics.

2. **Causal + not controllable** → architectural review. The instability direction exists but cannot be reached from current inputs. Add circuit breakers, decouple the dependency, or introduce new control surfaces.

3. **Symptomatic + lagging** → post-incident analysis. These features are useful for root-cause analysis *after* an incident, not for prevention. Do not alert on them — they will fire late and create noise.

4. **Latent risk** → canary monitoring. These features point along instability directions that are not currently active. Monitor their activation trend; a rising baseline suggests the system is drifting toward a failure mode it hasn't yet entered.

---

## 9. Mathematical Foundation: Why This Works

The empirical results above are not coincidence. The pipeline's diagnostic power follows from a chain of mathematical arguments, each grounded in established theory.

### 9.1 Koopman Approximation Guarantees

If the learned latent dynamics approximate Koopman eigenfunctions — and the evidence from deep Koopman learning [Lusch et al., 2018; Brunton et al., 2022] suggests they do — then the Jacobian eigenvalues of the learned transition approximate the *true* system's Koopman eigenvalues in the finite-dimensional subspace spanned by the learned coordinates. The spectral radius is not a model artifact; it is an estimate of the system's actual contraction/expansion rate.

The approximation quality depends on (a) the richness of the latent space (higher $d$ → more eigenfunctions captured) and (b) the prediction accuracy (low reconstruction loss → the learned dynamics faithfully represent the real dynamics). Both are verifiable from training metrics.

### 9.2 SAE Features as Koopman Modes

Koopman mode decomposition expresses observables as linear combinations of eigenfunctions:

$$x_t = \sum_k c_k \lambda_k^t \phi_k(x_0)$$

The SAE decomposes the latent representation — which approximates Koopman eigenfunction coordinates — into sparse directions. If the SAE features align with the Koopman modes, they provide an interpretable decomposition of the system's temporal evolution into independently-evolving components. The auto-labeling step (correlating features with observables and modes) recovers the physical meaning of each mode.

### 9.3 Controllability and Reachability

The controllability Gramian directly extends classical linear systems theory [Kalman, 1960] to the mode-conditioned, locally-linearized setting. The interpretation is standard: $\mathcal{C}_m$ characterizes the reachable set from observation-space inputs under mode $m$'s average dynamics. When combined with SAE feature projection, we obtain a per-feature, per-mode controllability score — answering, for each candidate intervention direction, whether the system's inputs can actually reach it.

### 9.4 The Key Synthesis

The power of the approach lies in the *combination* of these analyses. Any one alone has limited value:

- Stability analysis without feature decomposition tells you *that* the system is unstable, not *why*.
- Feature-stability correlation without eigenvector alignment conflates causes with symptoms.
- Causal identification without controllability analysis identifies root causes you may not be able to address.
- Controllability without lead-lag analysis identifies actionable directions but with no warning time.

The pipeline integrates all four, using the model's learned weights as the common substrate from which each analysis is derived. The weights encode the theory; the Jacobian, eigenstructure, SAE features, and Gramians are different *readings* of that theory.

---

## 10. Limitations and Future Directions

### What This Analysis Cannot Do

**Synthetic data, not production telemetry.** All four systems use simulated telemetry with known ground-truth dynamics. The thermostat+fault simulation captures the *qualitative* structure of microservice dynamics (normal homeostasis, fault-mode positive feedback, cascading propagation) but it does not capture the full complexity of production systems: non-stationary baselines, external traffic patterns, deployment changes, multi-tenancy effects. Applying the pipeline to real telemetry is the critical next step, and we expect the training to be harder (noisier data, more modes, partial observability) and the analysis to require more careful calibration.

**Stationarity assumption.** The SSM assumes that the transition dynamics within each mode are stationary — the same failure mode always produces the same dynamical signature. Real systems evolve: services are updated, infrastructure changes, and failure modes mutate. The current pipeline would need periodic retraining or online adaptation to track non-stationary systems.

**Bounded system scope.** The pipeline is designed for *bounded* systems — known topology, enumerable failure modes, finite observable set. It is not a foundation model. Each analysis is specific to the system it was fitted to, and the features, stability maps, and causal decompositions do not transfer between systems. This is a feature, not a bug — the theory is specific because the *system* is specific — but it means the pipeline must be run independently for each system of interest.

**Linear approximation at critical points.** The Jacobian-based stability analysis is a *local* linearization. At operating points where the dynamics are strongly nonlinear, the spectral radius of the Jacobian may not capture the full stability picture — nonlinear effects like limit cycles, strange attractors, or bifurcations require higher-order analysis. The current pipeline uses the Jacobian as a first-order diagnostic, which is effective for the systems studied here but may miss qualitative phenomena in more complex dynamics.

**Feature labeling incompleteness.** Some SAE features remain unlabeled — they encode compound patterns in latent space that don't map cleanly to individual observables or modes. These features are often among the most important (the top-ranked feature in both IoT and financial systems is unlabeled). This is not a failure — the model learned something real that doesn't decompose into single-metric terms — but it limits the immediate interpretability of the diagnostic output.

### What Comes Next

**Production validation.** The immediate priority is fitting the pipeline to production telemetry from real microservice systems and validating whether the learned stability maps, causal categories, and lead-lag indicators correspond to actual incident patterns. This requires (a) data pipelines from APM/observability platforms, (b) incident timeline correlation, and (c) careful handling of mode labeling in the absence of ground truth.

**Online adaptation.** Extending the SSM with continual learning — updating the transition function as the system evolves — would address the stationarity limitation. Approaches from meta-learning [Finn et al., 2017] and online Koopman learning [Zhang et al., 2019] are natural starting points.

**Nonlinear stability analysis.** Beyond the Jacobian's local linearization, computing Lyapunov functions or characterizing the spectrum of the full Koopman operator (not just the finite-dimensional approximation) would provide global stability guarantees. Recent work on data-driven Lyapunov functions [Richards et al., 2018] and extended dynamic mode decomposition (EDMD) [Williams et al., 2015] suggests tractable paths forward.

**Multi-system comparison.** While the current analysis treats each system independently, comparing the *structure* of learned theories across systems — are there universal stability patterns? Do all microservice systems exhibit the same causal-to-symptomatic ratio scaling? — could reveal domain-level insights beyond any single system.

**Interactive diagnostics.** The feature cards and stability maps are currently static reports. An interactive tool that allows engineers to query the model — "what happens if I add a circuit breaker between these two services?" or "which features become causal under a mode I haven't seen yet?" — would realize the full potential of the "model as instrument" framing.

---

## 11. The Theory in the Weights

Let us revisit what we have actually done.

We trained four neural state-space models on simulated telemetry from four different microservice ecosystems. We then treated each trained model not as a prediction engine to be queried on future inputs, but as a *diagnostic instrument* to be analyzed — extracting from its internal structure a comprehensive picture of each system's stability, causal dynamics, and intervention landscape.

Every readout in this analysis came from the model's weights and their derivatives:

- The **Jacobian eigenvalues** are the model's theory about which directions amplify perturbations and which suppress them.
- The **SAE features** are the model's theory about which observable combinations recur and co-activate.
- The **eigenvector alignment** tests whether those recurring patterns point in the direction the system is actually being destabilized.
- The **ablation rescue scores** provide a causal test — zeroing a feature and measuring the change in stability is a controlled experiment on the model's internal representation.
- The **controllability Gramians** characterize, for each mode, which latent directions are reachable from available inputs — a measurement of the system's *theoretical* intervention surface.

None of this required knowing the system's true equations. The models discovered the dynamics from data and encoded them in parameters. The parameters *are* the theory.

The finding that instability is **emergent** — every component individually healthy, yet the coupled system exhibits $\rho > 1$ — is visible only because the model learned the complete transition function, including inter-component coupling. Standard per-service monitoring, which decomposes the system into independent components, is structurally incapable of detecting emergent instability. The trained model sees it because it was never told to decompose — it learned the system as a whole.

The finding that **most destabilizing features are symptomatic, not causal** — and that this ratio worsens with system complexity — is a structural prediction of the pipeline, validated across four systems. It explains why conventional threshold-based alerting, which treats every anomalous metric as a potential root cause, generates alarm fatigue in complex systems: it cannot distinguish the causal few from the symptomatic many. The model's internal geometry — specifically, the alignment between SAE features and Jacobian eigenvectors — provides exactly this distinction.

The trained model is not an opaque prediction machine. It is a structured, analyzable, mechanistic representation of the system it was fitted to. Its weights are not just numbers that happen to produce good forecasts — they are a compressed, executable theory of the system's dynamics. And that theory, once extracted, provides a compressed readout of the system's architecture — the kind of structural insight that complements raw monitoring with something qualitatively different.

---

## 12. What Comes Next: Hearing Structure

Part 1 argued that weights encode theory. This essay demonstrated it: four systems, four trained models, four sets of mechanistic diagnostics — each extracted entirely from the internal structure of the learned parameters.

But the systems studied here, for all their operational realism, share a limitation: their structure must be *read* — inspected numerically, visualized in plots, interpreted through tables. The diagnostics are quantitative and actionable, but they are not *perceptible*. You cannot see a Jacobian eigenvalue or hear a controllability Gramian.

Part 3 applies the same "weights as theory" framework to a domain where the learned structure can be *heard*. When a neural model is trained on audio — specifically, when it learns to map between tuning systems, collapsing 12-tone equal temperament into 5-TET and attempting to reconstruct the original — the theory it encodes in its weights has a direct perceptual correlate. The spectral decomposition of harmonic structure, the algebraic constraints of just intonation and equal temperament, the geometry of consonance and dissonance — all of these are properties of the learned weights that manifest as *audible* differences in the output.

Where this essay derived stability and causality from a trained model's Jacobian, the next essay will derive harmonic structure and compositional geometry from a trained model's spectral representations. The same mathematical framework — Koopman eigenfunctions, sparse decomposition, geometric alignment — applies. The difference is that the Koopman modes of a harmonic system *are* the overtone series, and the theory in the weights can be verified by listening.

---

## References

- Bricken, T., Templeton, A., Batson, J., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Anthropic.
- Brunton, S. L., Budišić, M., Kaiser, E., & Kutz, J. N. (2022). *Modern Koopman Theory for Dynamical Systems.* SIAM Review, 64(2), 229–340.
- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS, 113(15), 3932–3937.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600.
- Elhage, N., Hume, T., Olsson, C., et al. (2022). *Toy Models of Superposition.* Anthropic.
- Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML 2017.
- Hašić, A. & Edalat, A. (2024). *On the Relationship between State-Space Models and Koopman Embeddings.* arXiv:2411.12940.
- Kalman, R. E. (1960). *On the general theory of control systems.* IFAC Proceedings Volumes, 1(1), 491–502.
- Koopman, B. O. (1931). *Hamiltonian Systems and Transformation in Hilbert Space.* PNAS, 17(5), 315–318.
- Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). *Deep learning for universal linear embeddings of nonlinear dynamics.* Nature Communications, 9, 4950.
- Meng, K., Bau, D., Mitchell, E., & Finn, C. (2023). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
- Mezić, I. (2005). *Spectral properties of dynamical systems, model reduction and decompositions.* Nonlinear Dynamics, 41, 309–325.
- Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). *Zoom In: An Introduction to Circuits.* Distill.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
- Richards, S. M., Berkenkamp, F., & Krause, A. (2018). *The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems.* CoRL 2018.
- Shwartz-Ziv, R. & Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information.* arXiv:1703.00810.
- Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). *A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition.* Journal of Nonlinear Science, 25, 1307–1346.
- Zhang, S., Bi, Y., & Gong, J. (2019). *Online Koopman Operator Learning for Autonomous Systems.* IEEE Transactions on Automatic Control.
- Zou, A., Phan, L., Chen, S., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.
