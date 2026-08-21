# Adaptive Modality Fusion: Mathematical Foundations

Date: 2026-07-31

Status: mathematical specification for research and implementation. The
definitions use confirmed 2025 competition behavior where stated and must be
versioned if official 2026 rules differ.

Companion document:
[Detailed Plan](ADAPTIVE_MODALITY_FUSION_DETAILED_PLAN.md).

## 1. Problem definition

Let:

- $q$ be a natural-language query;
- $t(q)$ be its task type;
- $\ell(q)$ be its language;
- $\mathcal M=\{v,c,o,a\}$ denote visual, caption, OCR, and ASR;
- $\mathcal D$ be the canonical frame corpus;
- $d\in\mathcal D$ be a candidate with immutable `frame_id`;
- $K$ be the retrieval depth per modality.

Each modality-specific retriever $R_m$ produces:

$$
L_m(q)
=
\left[
d_{m,1},d_{m,2},\ldots,d_{m,K}
\right].
$$

The rank of candidate $d$ in modality $m$ is:

$$
r_m(q,d)
=
\begin{cases}
i, & d=d_{m,i},\\
\infty, & d\notin L_m(q).
\end{cases}
$$

The fusion candidate pool is the exact identity union:

$$
C_q
=
\bigcup_{m\in\mathcal M}L_m(q).
$$

The union operation may merge scores and ranks for an identical `frame_id`,
but it must not infer or rewrite canonical frame metadata.

## 2. Static weighted reciprocal-rank fusion

For RRF constant $k>0$, define:

$$
x_m(q,d)
=
\begin{cases}
\dfrac{1}{k+r_m(q,d)}, & d\in L_m(q),\\[6pt]
0, & d\notin L_m(q).
\end{cases}
$$

Let:

$$
\mathbf x(q,d)
=
\left[
x_v(q,d),
x_c(q,d),
x_o(q,d),
x_a(q,d)
\right]^\top.
$$

Static task-weighted RRF is:

$$
s_{\mathrm{static}}(q,d)
=
\bar{\mathbf w}_{t(q)}^\top\mathbf x(q,d).
$$

The current neutral baseline is:

$$
\bar{\mathbf w}_{t}
=
[1,1,1,1]^\top.
$$

For any scalar $\alpha>0$:

$$
\operatorname{rank}
\left(
\alpha\bar{\mathbf w}_{t}^\top\mathbf x
\right)
=
\operatorname{rank}
\left(
\bar{\mathbf w}_{t}^\top\mathbf x
\right).
$$

Therefore only relative weights are identifiable from rankings.

## 3. Query-adaptive fusion

The goal is to learn:

$$
f_\theta(q,t,\ell,\mathcal L_q)
\rightarrow
\mathbf w(q),
$$

where:

$$
\mathcal L_q
=
\{L_m(q):m\in\mathcal M\}.
$$

The adaptive fusion score is:

$$
s_\theta(q,d)
=
\mathbf w(q)^\top\mathbf x(q,d).
$$

To resolve scale ambiguity, place the weights on the simplex:

$$
\Delta^{M-1}
=
\left\{
\mathbf w\in\mathbb R^M:
w_m\geq0,\ 
\sum_{m=1}^{M}w_m=1
\right\}.
$$

## 4. Positive weight floor

A pure softmax can assign a source a near-zero value. Define a positive floor:

$$
\mathbf p(q)
=
\operatorname{softmax}
\left(
\frac{\mathbf z(q)}{T}
\right),
$$

$$
w_m(q)
=
\epsilon
+
(1-M\epsilon)p_m(q),
$$

subject to:

$$
0\leq\epsilon<\frac{1}{M}.
$$

Then:

$$
w_m(q)\geq\epsilon,
\qquad
\sum_mw_m(q)=1.
$$

Here $T>0$ controls weight sharpness:

- $T<1$ produces sharper weights;
- $T>1$ produces more uniform weights.

Both $T$ and $\epsilon$ are validation hyperparameters.

## 5. Query representation

Let $E(q)\in\mathbb R^{d_e}$ be a frozen multilingual query embedding.
Let:

- $e_t\in\mathbb R^{d_t}$ be the task embedding;
- $e_\ell\in\mathbb R^{d_\ell}$ be the language embedding;
- $\phi(q)\in\mathbb R^{d_s}$ be deterministic surface features.

The complete query representation is:

$$
\mathbf h_q
=
\left[
E(q);
e_{t(q)};
e_{\ell(q)};
\phi(q)
\right].
$$

### 5.1 Linear gate

$$
\mathbf z_{\mathrm{pre}}(q)
=
W\mathbf h_q+\mathbf b.
$$

This model tests whether modality utility is approximately linearly separable
in the frozen query space.

### 5.2 One-hidden-layer gate

$$
\mathbf g_q
=
\operatorname{GELU}(W_1\mathbf h_q+\mathbf b_1),
$$

$$
\mathbf z_{\mathrm{pre}}(q)
=
W_2\operatorname{Dropout}(\mathbf g_q)+\mathbf b_2.
$$

The MLP is justified only if it improves held-out retrieval utility over the
linear gate.

## 6. Retrieval-list reliability

Query intent is not sufficient to estimate retrieval quality. Define
post-retrieval diagnostics for each modality.

Let the sorted raw scores returned by modality $m$ be:

$$
s_{m,1}\geq s_{m,2}\geq\cdots\geq s_{m,K}.
$$

### 6.1 Source-specific calibration

Raw scores from different embedding spaces are not comparable. Fit source
statistics on the training fold:

$$
\mu_m
=
\mathbb E_{\mathrm{train}}[s_m],
\qquad
\sigma_m^2
=
\operatorname{Var}_{\mathrm{train}}[s_m].
$$

Then:

$$
\hat s_{m,i}
=
\frac{s_{m,i}-\mu_m}
{\max(\sigma_m,\delta)}.
$$

$\delta>0$ prevents division by zero.

Alternative monotonic calibration methods, such as empirical percentile
transforms, may be evaluated. Calibration must never use validation or test
judgments.

### 6.2 Top-score separation

$$
\Delta_m(q)
=
\hat s_{m,1}
-
\frac{1}{h-1}
\sum_{i=2}^{h}\hat s_{m,i}.
$$

For example, $h=5$ compares the top result with the next four. The value of
$h$ must be selected on the training/validation protocol.

### 6.3 Normalized entropy

Define:

$$
p_{m,i}
=
\frac{\exp(\hat s_{m,i}/\tau_s)}
{\sum_{j=1}^{K}\exp(\hat s_{m,j}/\tau_s)}.
$$

The normalized entropy is:

$$
H_m(q)
=
-\frac{1}{\log K}
\sum_{i=1}^{K}p_{m,i}\log p_{m,i}.
$$

$$
0\leq H_m(q)\leq1.
$$

Low entropy means that the modality concentrates score mass on fewer results.
It does not prove correctness, so it must be evaluated jointly with ground
truth.

### 6.4 Score-curve area

Min-max normalize within one result list:

$$
\tilde s_{m,i}
=
\frac{
\hat s_{m,i}-\hat s_{m,K}
}{
\max(\hat s_{m,1}-\hat s_{m,K},\delta)
}.
$$

Define area:

$$
A_m(q)
=
\frac{1}{K}
\sum_{i=1}^{K}\tilde s_{m,i}.
$$

Define an early-to-tail slope:

$$
G_m(q)
=
\frac{
\tilde s_{m,1}-\tilde s_{m,h}
}{
h-1
}.
$$

These summarize the ranked score curve. Their relationship with correctness
must be learned rather than assumed.

### 6.5 Source coverage

If a sparse source returns $n_m(q)\leq K$ usable candidates:

$$
P_m(q)
=
\frac{n_m(q)}{K}.
$$

### 6.6 Video concentration

Let $\mathcal V_m(q)$ be the unique videos in $L_m(q)$, and:

$$
\pi_{m,v}(q)
=
\frac{
\left|
\{d\in L_m(q):\operatorname{video}(d)=v\}
\right|
}{
|L_m(q)|
}.
$$

Define normalized video entropy:

$$
H_m^{\mathrm{video}}(q)
=
-\frac{
\sum_{v\in\mathcal V_m(q)}
\pi_{m,v}(q)\log\pi_{m,v}(q)
}{
\log|\mathcal V_m(q)|
}.
$$

Define concentration:

$$
V_m(q)
=
1-H_m^{\mathrm{video}}(q).
$$

The single-video edge case is defined as $V_m(q)=1$.

### 6.7 Pairwise ranked-list agreement

For modalities $m\neq n$:

$$
C_{m,n}(q)
=
\sum_{d\in L_m(q)\cap L_n(q)}
\frac{1}
{(k+r_m(q,d))(k+r_n(q,d))}.
$$

This gives more importance to agreement near the top of both lists.

Per-modality consensus is:

$$
C_m(q)
=
\frac{1}{M-1}
\sum_{n\neq m}C_{m,n}(q).
$$

Agreement can indicate complementary confirmation, but it can also reflect
duplicated captions, subtitles, or OCR text. It must be included in redundancy
ablations.

## 7. Hybrid gate

Let:

$$
\mathbf u(q)
=
\left[
\{\Delta_m,H_m,A_m,G_m,P_m,V_m,C_m\}_{m\in\mathcal M};
\{C_{m,n}\}_{m<n}
\right].
$$

A residual post-retrieval model produces:

$$
\mathbf z_{\mathrm{post}}(q)
=
g_\psi(\mathbf u(q)).
$$

The final logits are:

$$
\mathbf z(q)
=
\mathbf z_{\mathrm{pre}}(q)
+
\mathbf z_{\mathrm{post}}(q).
$$

The residual form has a useful interpretation:

- $\mathbf z_{\mathrm{pre}}$ estimates expected modality utility from intent;
- $\mathbf z_{\mathrm{post}}$ corrects that expectation using observed
  retrieval behavior.

## 8. Ground-truth relevance

### 8.1 Textual KIS

For ground-truth video $v_q^*$ and accepted interval $[s_q,e_q]$:

$$
y_q(d)
=
\mathbb 1
\left[
\operatorname{video}(d)=v_q^*
\land
s_q\leq\operatorname{frame\_idx}(d)\leq e_q
\right].
$$

The accepted set is:

$$
P_q
=
\{d\in C_q:y_q(d)=1\}.
$$

### 8.2 VQA

For submitted row $(d,a)$:

$$
R_q(d,a)
=
\mathbb 1[\operatorname{video}(d)=v_q^*]
\cdot
\mathbb 1[s_q\leq\operatorname{frame\_idx}(d)\leq e_q]
\cdot
\mathbb 1[\operatorname{AnswerCorrect}(a,a_q^*)].
$$

Frame-retrieval gate training may use the first two terms. Full VQA evaluation
must also include answer correctness and preserve the exact submitted answer
string.

### 8.3 TRAKE

Let the ground truth contain $N$ accepted event intervals:

$$
\mathcal I_q
=
\{[s_{q,j},e_{q,j}]\}_{j=1}^{N}.
$$

For a predicted row:

$$
\hat y
=
(v,f_1,\ldots,f_N),
$$

the 2025-style row score is:

$$
R_q(\hat y)
=
\mathbb 1[v=v_q^*]
\frac{1}{N}
\sum_{j=1}^{N}
\mathbb 1[
s_{q,j}\leq f_j\leq e_{q,j}
].
$$

The predicted frames must satisfy:

$$
f_1<f_2<\cdots<f_N
$$

and all frames must belong to the same predicted video.

### 8.4 Conversational KIS

Let $S_{t-1}$ be deterministic conversation state and $q_t$ the new turn:

$$
q_t'
=
\operatorname{Resolve}(q_t,S_{t-1})
$$

only when the turn is genuinely context dependent.

The gate receives:

$$
\mathbf w_t
=
f_\theta(q_t').
$$

Feedback-only turns update $S_t$ without generative resolution or new
retrieval.

## 9. Candidate-pool ceiling

Adaptive fusion cannot retrieve a positive candidate that is absent from every
source list.

Define candidate-union success:

$$
\operatorname{UnionHit}(q)
=
\mathbb 1[P_q\neq\varnothing].
$$

Candidate-union recall is:

$$
\operatorname{UnionRecall}
=
\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}
\operatorname{UnionHit}(q).
$$

If $P_q=\varnothing$, the query is an upstream candidate-generation failure.
It must not be converted into an all-negative router training example.

## 10. Training objectives

### 10.1 Multi-positive listwise objective

For temperature $\tau_r>0$:

$$
\mathcal L_{\mathrm{list}}(q)
=
\log
\sum_{d\in C_q}
\exp\left(\frac{s_\theta(q,d)}{\tau_r}\right)
-
\log
\sum_{d\in P_q}
\exp\left(\frac{s_\theta(q,d)}{\tau_r}\right).
$$

This objective increases the total probability mass assigned to accepted
frames.

### 10.2 Pairwise objective

For positive $p\in P_q$ and negative $n\in C_q\setminus P_q$:

$$
\mathcal L_{\mathrm{pair}}(q)
=
\sum_{p,n}
\lambda_{p,n}
\log
\left(
1+\exp(s_\theta(q,n)-s_\theta(q,p))
\right).
$$

$\lambda_{p,n}$ may approximate the change in competition utility caused by
swapping $p$ and $n$. This is a later experiment; listwise training is the
initial objective.

### 10.3 Task-prior regularization

Let $\bar{\mathbf w}_{t(q)}$ be the tuned task-static prior:

$$
\mathcal L_{\mathrm{prior}}(q)
=
D_{\mathrm{KL}}
\left(
\mathbf w(q)
\parallel
\bar{\mathbf w}_{t(q)}
\right).
$$

This discourages extreme per-query deviations when training evidence is
limited.

### 10.4 Translation consistency

For a verified translation pair $(q_{\mathrm{vi}},q_{\mathrm{en}})$:

$$
\mathcal L_{\mathrm{lang}}
=
\left\|
\mathbf w(q_{\mathrm{vi}})
-
\mathbf w(q_{\mathrm{en}})
\right\|_2^2.
$$

This loss must not be applied to unverified machine translations that change
query meaning.

### 10.5 Combined loss

$$
\mathcal L
=
\sum_{q\in\mathcal Q_{\mathrm{train}}}
\mathcal L_{\mathrm{list}}(q)
+
\lambda_p\mathcal L_{\mathrm{prior}}(q)
+
\lambda_\ell\mathcal L_{\mathrm{lang}}(q)
+
\lambda_2\|\theta\|_2^2.
$$

The initial experiment sets optional regularizers to zero, then adds them one
at a time through ablation.

## 11. Exact ranking utility

For a ranked set of submitted rows with task-specific row scores
$R_{q,1},R_{q,2},\ldots$, define:

$$
\operatorname{TopR}_q(k)
=
\max_{1\leq i\leq k}R_{q,i}.
$$

The confirmed 2025 query score is:

$$
U(q)
=
\frac{1}{5}
\sum_{k\in\{1,5,20,50,100\}}
\operatorname{TopR}_q(k).
$$

The dataset score is:

$$
U(\mathcal Q)
=
\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}U(q).
$$

This metric is discontinuous. The differentiable ranking loss trains the
model, while exact $U(\mathcal Q)$ selects configurations and checkpoints.

### 11.1 KIS simplification

If the first correct KIS row occurs at rank $r_q$:

$$
U_{\mathrm{KIS}}(q)
=
\frac{1}{5}
\sum_{k\in\{1,5,20,50,100\}}
\mathbb 1[r_q\leq k].
$$

This reveals why all five cutoff regions matter. Moving a correct result from
rank 21 to rank 20 changes the score even when Recall@1 and Recall@5 do not.

## 12. Oracle weights and regret

The per-query oracle is:

$$
\mathbf w_q^*
\in
\arg\max_{\mathbf w\in\Delta^{M-1}}
U\left(
q;
\operatorname{sort}_{d\in C_q}
\mathbf w^\top\mathbf x(q,d)
\right).
$$

Because ranking is piecewise constant in $\mathbf w$, the oracle may be
non-unique. It is primarily an upper bound, not a uniquely valid supervision
target.

Approximate the oracle with:

- simplex vertices;
- equal and task-static weights;
- grid samples;
- Dirichlet samples;
- local refinement around strong samples.

Define adaptive fusion regret:

$$
\operatorname{Regret}(q)
=
U(q;\mathbf w_q^*)
-
U(q;\mathbf w_\theta(q)).
$$

Mean regret is:

$$
\overline{\operatorname{Regret}}
=
\frac{1}{|\mathcal Q|}
\sum_q\operatorname{Regret}(q).
$$

The oracle-static gap estimates available headroom. The oracle-model gap
estimates the remaining weight-prediction problem.

## 13. TRAKE event-level adaptive fusion

Let the TRAKE query be decomposed into ordered event queries:

$$
q
\rightarrow
(q_1,q_2,\ldots,q_N).
$$

Each event receives its own weights:

$$
\mathbf w_j
=
f_\theta(q_j).
$$

Its candidate score is:

$$
s_j(d)
=
\mathbf w_j^\top\mathbf x(q_j,d).
$$

For one video $v$ and ordered frames
$\mathbf f=(f_1,\ldots,f_N)$, define:

$$
S(v,\mathbf f)
=
\sum_{j=1}^{N}s_j(f_j)
+
\eta
\sum_{j=2}^{N}
T(f_{j-1},f_j),
$$

subject to:

$$
\operatorname{video}(f_j)=v
\quad\forall j,
$$

$$
\operatorname{frame\_idx}(f_1)
<
\cdots
<
\operatorname{frame\_idx}(f_N).
$$

$T$ is a transition term that may penalize implausible gaps or reward
coherent temporal windows. It must be validated separately from modality
weighting.

The optimal sequence is:

$$
(v^*,\mathbf f^*)
=
\arg\max_{v,\mathbf f}S(v,\mathbf f).
$$

Dynamic programming or beam search may solve this constrained alignment after
per-event candidate generation.

## 14. Optional hard routing

Continuous fusion queries all indexes. Let:

$$
z_m(q)\in\{0,1\}
$$

indicate whether modality $m$ is searched, and $c_m$ be its measured
online cost.

Expected search cost is:

$$
C(q)
=
\sum_m c_mz_m(q).
$$

A cost-aware training or selection objective is:

$$
\mathcal J
=
\mathcal L_{\mathrm{rank}}
+
\lambda_c
\mathbb E_q[C(q)].
$$

Hard routing introduces a new failure mode:

$$
z_m(q)=0
$$

may remove the only modality containing an accepted candidate. Therefore it
must be evaluated as an accuracy-latency Pareto problem, not assumed superior
because it uses fewer searches.

One conservative selection rule is:

$$
z_m(q)
=
\mathbb 1[
w_m^{\mathrm{pre}}(q)\geq\tau
]
\lor
\mathbb 1[
m=\arg\max_j w_j^{\mathrm{pre}}(q)
].
$$

The second term ensures that at least one modality is searched.

## 15. Statistical evaluation

For method difference per query:

$$
\Delta_q
=
U_A(q)-U_B(q).
$$

Report:

$$
\bar\Delta
=
\frac{1}{|\mathcal Q|}
\sum_q\Delta_q.
$$

Use paired bootstrap resampling over queries:

1. sample $|\mathcal Q|$ query IDs with replacement;
2. compute $\bar\Delta_b$;
3. repeat for many bootstrap samples;
4. report a percentile confidence interval.

For neural models, also report variation over random seeds. Query bootstrap
and seed variation measure different uncertainty sources and should not be
silently combined.

## 16. Identifiability and interpretation

The predicted values are fusion coefficients, not probabilities that a
modality is correct.

Several issues prevent a direct probabilistic interpretation:

- multiple weight vectors can produce the same ranking;
- modality evidence is correlated;
- caption may duplicate visual or OCR information;
- subtitles can make OCR and ASR redundant;
- a high source score can still be confidently wrong.

Interpretability should therefore be supported by:

- held-out utility;
- counterfactual removal of each source;
- oracle regret;
- calibration and corruption experiments;
- per-query failure analysis.

## 17. Inference invariants

For every query:

$$
\mathbf w(q)\in\Delta^{3},
$$

$$
w_m(q)\geq\epsilon,
$$

$$
\mathbf w(q)\ \text{contains only finite values}.
$$

For every candidate:

$$
\operatorname{Identity}_{\mathrm{after\ fusion}}(d)
=
\operatorname{Identity}_{\mathrm{before\ fusion}}(d).
$$

If inference fails:

$$
\mathbf w(q)
\leftarrow
\bar{\mathbf w}_{t(q)}.
$$

The fallback is deterministic and selected from held-out benchmark evidence.

## 18. Model-selection hierarchy

Select the simplest model satisfying the metric and latency evidence:

1. equal RRF;
2. tuned task-static RRF;
3. linear query gate;
4. query MLP;
5. evidence-only gate;
6. query plus evidence gate;
7. optional hard routing.

A more complex method is promoted only when it improves exact held-out utility
and the gain remains defensible under ablation, latency measurement, and
uncertainty analysis.

## 19. Primary references

- [Query-Adaptive Fusion for Multimodal Search](https://research.google/pubs/query-adaptive-fusion-for-multimodal-search/)
- [Learning a Text-Video Embedding from Incomplete and Heterogeneous Data](https://arxiv.org/abs/1804.02516)
- [Multi-modal Transformer for Video Retrieval](https://www.ecva.net/papers/eccv_2020/papers/123490205.pdf)
- [Query-Adaptive Late Fusion for Image Search and Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2015/html/Zheng_Query-Adaptive_Late_Fusion_2015_CVPR_paper.html)
- [Query-Adaptive Late Fusion for Hierarchical Fine-Grained Video-Text Retrieval](https://ieeexplore.ieee.org/document/9927461/)
- [MMMORRF](https://arxiv.org/abs/2503.20698)
- [Smart Routing for Multimodal Video Retrieval](https://openaccess.thecvf.com/content/ICCV2025W/MRR%202025/html/Dela_Rosa_Smart_Routing_for_Multimodal_Video_Retrieval_When_to_Search_What_ICCVW_2025_paper.html)
- [Mixture of Retrievers](https://aclanthology.org/2025.emnlp-main.601/)
