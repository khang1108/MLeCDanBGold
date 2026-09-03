# Research Roadmap for Temporal Multimodal Retrieval in HCMAI: From a Strong Competition System to a Defensible SoICT Paper

## Executive diagnosis

After inspecting `src_hcmai_v7.zip` and comparing the current system against recent work in temporal video retrieval, temporal grounding, long-video reasoning, sequence alignment, and the papers emerging directly from AI Challenge HCMC 2025, my main conclusion is:

> **Your strongest paper should not be “another multimodal retrieval system with Dynamic Programming.” It should be a paper about structured temporal reasoning over multi-event queries, where Dynamic Programming is only the inference algorithm.**

That distinction matters because the novelty space around “multimodal retrieval + DP” is already crowded. DANTE explicitly proposes dynamic programming for TRAKE; Lucifer-TRACE combines DP-style temporal search with LVLM verification; MADTempo aggregates sequential event evidence; U-CESE introduces clip-based retrieval and temporally consistent captions; and another AIC 2025 system already combines query expansion with cross-modal temporal event retrieval. citeturn12academia1turn11academia17turn13academia24turn12academia0turn14search11

The most promising gap in your current code is considerably deeper:

> **The current DP enforces temporal order, but it does not actually model the semantic transition between consecutive events.**

More importantly, there is a mathematical issue in the present objective that makes this distinction very concrete. Your current objective is documented as

$$
\mathcal{S}(p_{1:M})
=
\sum_{i=1}^{M} U_i(p_i)
-
\lambda
\sum_{i=2}^{M}
\left(t_{p_i}-t_{p_{i-1}}\right),
$$

under the strict chronological constraint

$$
p_1 < p_2 < \cdots < p_M.
$$

Because timestamps are monotonically increasing, the temporal term **telescopes**:

$$
\sum_{i=2}^{M}
(t_{p_i}-t_{p_{i-1}})
=
t_{p_M}-t_{p_1}.
$$

Therefore,

$$
\mathcal{S}
=
\sum_i U_i(p_i)
-
\lambda(t_{p_M}-t_{p_1}).
$$

So your current temporal penalty does **not distinguish the internal pacing of the sequence**.

For example, ignoring unary scores, these paths receive exactly the same temporal penalty:

$$
[0s,1s,100s]
$$

and

$$
[0s,99s,100s].
$$

Both span 100 seconds.

That means the DP currently knows:

> event A must occur before B, which must occur before C, and the entire sequence should preferably be compact.

It does **not** know:

> how A should transition into B; how long that transition should plausibly take; whether it is the same person/object; whether an object changes state; whether B is the consequence of A; whether the motion between the two anchors is compatible with the query; or whether one query clause should be skipped because it is not visually groundable.

This is exactly where I would build the paper.

A second important strategic point is the publication landscape. The official AI Challenge HCMC 2026 page describes the task as intelligent multimedia retrieval, explicitly mentions conventional and automated competition modes, and states that strong methods may be selected for a Lifelog and Multimedia Event Retrieval special session at SoICT 2026. citeturn11search3 The general SoICT submission site currently lists multimedia intelligence, multimedia information retrieval, event retrieval, multimodal lifelog retrieval, and event understanding as in-scope topics. citeturn11search2 There is currently an inconsistency between official pages concerning whether the special-session proceedings route is ACM or the general SoICT CCIS proceedings, so I would treat the competition invitation route as administratively separate and **not let the normal SoICT submission timeline constrain the scientific plan**, as you requested. citeturn11search3turn11search0

My recommendation, therefore, is to formulate the project around:

> **Transition-Aware Multimodal Alignment for Multi-Event Video Retrieval**

with three tightly connected ideas:

**structured event-query understanding → adaptive multimodal unary evidence → event-conditioned pairwise temporal transitions**, decoded through an efficient candidate-lattice DP.

A coarse-to-fine clip refinement stage should support this method. Null/skip states should make it robust. A VLM verifier can be included, but only as a secondary module rather than the paper's central novelty.

This gives you a paper that is simultaneously:

- directly useful for KIS and TRAKE;
- compatible with your current architecture instead of requiring a rewrite;
- scientifically distinguishable from DANTE;
- much more defensible than “we changed the embedding model”;
- experimentally decomposable into clean hypotheses and ablations;
- compatible with an automated retrieval setting, because the expensive reasoning can be restricted to a small candidate lattice.

I also checked the research-plugin inventory available in this environment. A scientific-search connector such as Consensus is discoverable but was not installed/connected in the session, so I did not make the report depend on it. The literature claims below are instead grounded primarily in official challenge pages and primary publication repositories including arXiv, CVF, ACL Anthology, PMLR, NeurIPS, and Springer.

## What the current system actually implements

The codebase is substantially better organized than a typical competition prototype. That is valuable for research because there is already a clean boundary between evidence generation, temporal decoding, task-specific projection, offline enrichment, and model inference.

The present architecture can be summarized as:

$$
\text{query}
\rightarrow
\text{event strings}
\rightarrow
\text{multimodal frame scores}
\rightarrow
\text{per-video score matrices}
\rightarrow
\text{monotonic DP}
\rightarrow
\text{KIS/TRAKE output}.
$$

### The current DP is an efficient ordered unary model, not a transition model

`src/hcmai/temporal/dp.py:78–165` performs strict monotonic alignment. Given an event-by-frame matrix, each event selects one later frame than the previous event.

The recurrence is implemented through the prefix-maximum transformation around `dp.py:120–142`. This avoids the naïve quadratic predecessor search and gives an essentially linear scan over the frames for each event:

$$
O(MF)
$$

for \(M\) query events and \(F\) frames in one video.

That is an excellent engineering baseline.

But scientifically, the model is essentially:

$$
\text{unary semantic compatibility}
+
\text{monotonicity}
+
\text{global compactness}.
$$

The repository's own `src/hcmai/temporal/README.md:63–76` correctly identifies several missing components: entity continuity, state transitions, multimodal dense alignment, multi-frame VLM verification, and incremental DP.

The important additional conclusion from inspecting the implementation is the telescoping property above. This makes **transition modeling** even more compelling than the README alone suggests.

### The temporal structure is entirely determined by independent frame emissions

Each event \(e_i\) produces a unary score for every frame:

$$
U_i(f)=
\operatorname{sim}(e_i,f).
$$

Nothing in the current score says:

$$
P(f_j \mid f_i,e_i,e_{i+1}),
$$

or asks whether the transition

$$
f_i \rightarrow f_j
$$

semantically realizes

$$
e_i \rightarrow e_{i+1}.
$$

This distinction becomes crucial for queries such as:

> a man walks toward a table → picks up a red cup → drinks from it → puts the same cup down.

Independent unary retrieval might find four excellent frames containing:

- a man near a table;
- somebody holding a red cup;
- somebody drinking;
- a red cup on a table.

Yet these frames could correspond to different people, different cups, different shots, or even different sub-stories inside a long news video.

Strict ordering alone cannot reject that path.

This is the scientific difference between **ordered retrieval** and **temporal event reasoning**.

### Current multimodal fusion is useful but globally fixed

The dense temporal scorer in `src/hcmai/retrieval/evidence/dense.py:42–63` computes three signals:

$$
S_{\text{visual}},\qquad
S_{\text{context}},\qquad
S_{\text{ASR}},
$$

with SigLIP2 for visual retrieval and BGE-M3 for textual evidence according to the pinned configuration in `thundercompute/config.yaml`.

Each event-by-corpus score row is independently min-max normalized, after which the three modalities are combined as

$$
S =
w_v S_v+
w_c S_c+
w_a S_a.
$$

The default configuration in `src/hcmai/common/config.py:312–328` is

$$
w_v=w_c=w_a=\frac13.
$$

When BM25 is enabled, the hybrid scorer again uses a fixed convex combination, defaulting to

$$
0.5S_{\text{dense}}+0.5S_{\text{BM25}}
$$

in `common/config.py:342–358`.

This produces a very good experimental opportunity.

Consider these two events:

> “The presenter says the words ‘artificial intelligence’.”

and

> “A motorcycle turns left and narrowly passes a car.”

The optimal evidence mixture is obviously unlikely to be identical. ASR is potentially decisive for the first; motion-aware visual evidence should dominate the second.

The current fusion does not express that distinction.

There is another calibration issue. Per-event, per-modality min-max normalization largely removes **absolute confidence**. A weak/noisy modality can still be stretched across \([0,1]\), so small meaningless score differences can become comparable to the dynamic range of a genuinely informative modality.

That suggests a second, relatively low-risk research component:

> **query-conditioned modality reliability rather than globally fixed modality weights.**

### FrameContext is multimodal but not temporal

`FrameContext V1` is built from

$$
[\text{CAPTION}],
[\text{VISIBLE\_TEXT}],
[\text{OBJECTS}]
$$

with token budgets 80/80/40 according to `offline/enrichment/context/config.py` and `serializer.py`.

ASR is deliberately maintained separately as timestamped evidence; the offline documentation explicitly says it is excluded from FrameContext.

This separation is architecturally sensible, but FrameContext remains a **same-frame description**.

It does not contain:

- previous-frame state;
- next-frame state;
- motion;
- object trajectory;
- persistent entity memory;
- shot-level context;
- “before/after” information.

The broader literature increasingly emphasizes precisely these temporal structures. LongVALE, for example, treats long videos as sequences of multimodal events and explicitly builds vision-audio-language events with temporal boundaries and relation-aware descriptions rather than independent frames. citeturn15search5 VideoStir likewise represents long video as a spatio-temporal clip graph and retrieves evidence through multi-hop structured reasoning rather than treating the video as a flat frame collection. citeturn15search6

### Query decomposition is syntactic rather than semantic

`src/hcmai/temporal/planner.py` splits KIS text deterministically using lines or sentence boundaries.

That is robust and reproducible, but “sentence” is not necessarily equivalent to “temporally groundable event.”

For example:

> “A woman wearing glasses enters the room, speaks to the seated man and afterwards gives him a document while another person watches.”

One sentence contains at least three useful temporal anchors:

$$
\text{enters}
\rightarrow
\text{speaks}
\rightarrow
\text{gives document}.
$$

Conversely:

> “The same person, who appeared earlier in the report, is still speaking.”

contains language that may encode one event plus an identity constraint rather than several independent retrieval events.

Your Qwen query-preparation subsystem already has structured output and can generate exactly five aligned paraphrase bundles (`query_preparation/service.py:64–89`), but the primary KIS/TRAKE workflows do not yet turn the original query into an explicit **event graph**.

ED-VTG shows that enriching grounding queries can be valuable, but it also explicitly trains with multiple-instance learning to select among query variants and suppress harmful hallucinated enrichments. citeturn15search0 That is an important warning: “LLM query expansion” alone should not be treated as automatically beneficial.

### Full-corpus scoring precedes temporal reasoning

The repository itself documents that there is no candidate-video shortlisting before the temporal decoder. Every selected visual-index frame is scored for each event.

For 873 videos, that may be entirely acceptable for inexpensive embedding similarity. Therefore, I would **not** pitch hierarchical retrieval primarily as a scalability contribution.

Its real purpose should be different:

> inexpensive global retrieval should create a small search lattice on which we can afford substantially richer temporal reasoning.

That changes the architecture from

$$
\text{expensive reasoning over everything}
$$

to

$$
\text{cheap recall}
\rightarrow
\text{rich temporal reasoning over plausible regions}.
$$

This coarse-to-fine principle is strongly supported by contemporary long-video work. ReVisionLLM first identifies broad relevant regions and recursively narrows them to precise temporal boundaries; VideoTree likewise constructs a query-adaptive hierarchical representation and progressively refines relevant video regions. citeturn15search1turn15search8

### The code already contains useful machinery for the proposed research

Several existing components mean the research proposal does not require throwing away your system.

Your code already has:

- SigLIP2 visual embeddings;
- BGE-M3 multimodal textual evidence;
- OCR;
- ASR;
- YOLOE detections;
- Qwen3-VL captioning;
- Qwen query preparation;
- a Qwen3-VL reranker;
- DINO embedding API support;
- shot/event preprocessing interfaces;
- canonical timestamps and frame identities;
- a clean DP module.

The Qwen reranker is particularly relevant. `thundercompute/README.md:108–110` correctly notes that it can only reorder retrieved candidates; it cannot recover missing candidates.

That naturally motivates an architecture in which VLM reasoning is used **after high-recall candidate construction**, not as the retrieval engine itself.

The KIS readout also deserves attention. `orchestration/workflows/kis.py:39–43` currently chooses an upper-middle event in the aligned path as the representative frame. That is deterministic, but not query-dependent. For an asymmetric event sequence, the discriminative KIS evidence might occur near the beginning or end of the inferred segment rather than at its midpoint.

This is a small but very measurable secondary problem.

## Literature and novelty landscape

The literature immediately surrounding this challenge has moved quickly. It is important to know which apparently attractive ideas are already “occupied.”

### Dynamic programming alone is no longer enough

The most direct competitor is **DANTE**, proposed in *Integrated Semantic and Temporal Alignment for Interactive Video Retrieval*. The paper is explicitly motivated by AI Challenge HCMC 2025 TRAKE and proposes Dynamic Alignment of Narrative Temporal Events using dynamic programming. citeturn12academia1

Therefore a paper whose principal claim is

> “we use DP to align multiple query events chronologically”

will be very difficult to position as novel.

Your DP can absolutely remain—it is a good inference engine—but the **energy/function being optimized needs to be new**.

There is an even stronger warning. **Lucifer-TRACE**, now listed in the Springer proceedings for SoICT 2025, combines dynamic-programming-based temporal search with LVLM semantic verification. citeturn14search11turn14search5

Consequently:

> **DP + Qwen verification is also not enough as the headline contribution.**

You can and probably should perform VLM verification, but it should support the main algorithm rather than define the novelty.

### Query augmentation is a crowded direction

QUEST uses an LLM for query rewriting and external image search to handle out-of-knowledge queries. citeturn12academia1

MADTempo combines multi-event temporal retrieval with external image search as an OOD fallback. citeturn11academia17

RAPID already frames LLM query correction/enrichment and parallel retrieval as a major component of HCMC video retrieval. citeturn12academia3

Another 2025 challenge system explicitly combines LLM query expansion with cross-modal temporal event retrieval. citeturn12academia0

So:

> do query parsing and controlled paraphrasing because they improve the system, but do not make “LLM query expansion” the central paper claim.

A more interesting query-side contribution is **structural parsing**:

$$
q
\rightarrow
(E_1,R_{12},E_2,R_{23},\ldots,E_M)
$$

where each event records entities, actions and states, and each \(R_{i,i+1}\) represents the expected transition.

That structure directly connects language understanding to your temporal algorithm.

### Keyframe extraction and temporal caption memory are also occupied

U-CESE proposes a unified clip-based search engine, DAKE keyframe extraction, and ReCap, a temporally consistent captioning framework. citeturn13academia24

The cross-modal temporal retrieval paper by Vo et al. also proposes adaptive keyframe selection through KDE-GMM thresholding. citeturn12academia0

Therefore I would not write a paper whose primary contribution is:

> “better keyframes”

or:

> “captions with memory.”

Both remain excellent supporting experiments, especially because your current FrameContext is temporally independent, but they are no longer clean novelty territory in this specific research community.

### Multi-event queries have a strong connection to sequence alignment research

The most useful conceptual shift is to think of your problem as a mixture of:

$$
\text{cross-modal retrieval}
+
\text{sequence alignment}
+
\text{temporal grounding}.
$$

Classic DTW already models monotonic sequence alignment. Soft-DTW makes the DTW objective differentiable so alignment can become a learning objective rather than only an inference operation. citeturn10search0

More relevant to your specific failure cases is **Drop-DTW**, which allows elements to be dropped while aligning the common signal between noisy sequences. It was specifically evaluated on temporal step localization and cross-modal retrieval/localization. citeturn10search1

That directly motivates allowing

$$
z_i=\varnothing
$$

for query events that are:

- abstract;
- redundant;
- incorrectly split;
- missing from available keyframes;
- badly paraphrased;
- audible but not visible;
- visually ambiguous.

Your current DP forces every event to select a frame. That creates a classic garbage-in-path problem: one bad event can drag the entire alignment toward the wrong region.

StepFormer is also informative. It uses order-aware supervision for discovering and localizing procedural steps while filtering irrelevant phrases and demonstrates zero-shot multi-step localization. citeturn10search2

Again, the lesson is that a multi-event query should not automatically imply

$$
\text{exactly one mandatory frame per text fragment}.
$$

### Modern grounding work argues strongly for clip-level temporal representations

The current system primarily reasons over keyframes, but actions are fundamentally temporal.

An image can often tell us:

> a person is holding a cup.

A short clip is much better for:

> the person picks up the cup.

Similarly:

> the car is beside the motorcycle

is static, while

> the car overtakes the motorcycle

is temporal.

Recent grounding methods increasingly address this explicitly. Sparse-Dense Side-Tuner shows strong temporal-grounding performance using InternVideo2 features while remaining parameter-efficient. citeturn15search13 LongVALE argues for fine-grained multimodal event understanding over vision, speech/audio, and temporal boundaries. citeturn15search5 TemporalVLM targets dense captioning, temporal grounding, highlight detection, and action segmentation under a unified long-video temporal representation. citeturn15search14

For your system, this suggests:

> keep image/keyframe embeddings for global retrieval; introduce true clip/motion representations only around candidate temporal regions.

That gives most of the benefits without turning the entire 873-video corpus into an expensive Video-LLM workload.

### Temporal direction should become an explicit stress test

ArrowGEV is particularly interesting conceptually because it distinguishes events whose semantics change when time is reversed from those that are insensitive to reversal, and explicitly trains temporal direction awareness. citeturn15search15

This suggests one of the cleanest evaluations your paper could introduce.

For a query:

$$
A\rightarrow B\rightarrow C,
$$

construct hard negatives containing:

$$
C\rightarrow B\rightarrow A
$$

or

$$
A\rightarrow C\rightarrow B.
$$

A frame-bag retrieval system may score these videos highly because all objects/actions are present.

A genuine temporal method should not.

This could become a particularly convincing experiment because it directly tests whether your method is learning/reasoning about chronology rather than merely benefiting from better semantic retrieval.

### The novelty map for your project

I would evaluate candidate directions as follows.

| Direction | Scientific novelty in this challenge landscape | Expected impact | Engineering cost | Recommendation |
|---|---:|---:|---:|---|
| Event-conditioned pairwise transition DP | **Very high** | **Very high** on TRAKE, likely KIS | Medium | **Main contribution** |
| Null/skip-event robust alignment | High | High on noisy queries | Low–medium | **Integrate into main method** |
| Query-conditioned modality gating | Medium–high | High and broad | Low–medium | **Strong secondary contribution** |
| Coarse-to-fine keyframe → clip refinement | High if tied to transition reasoning | High for actions/motion | Medium | **Strong secondary component** |
| Entity/state continuity | **Very high** | High on difficult TRAKE | Medium–high | **Excellent extension / strongest version** |
| VLM path verification | Medium | Moderate–high | Medium | Use, but not headline |
| Query expansion | Low–medium | Often useful | Low | Engineering only |
| Better keyframe extraction | Medium | Potentially high | Medium | Supporting experiment |
| Temporally consistent captioning | Medium | High for semantic context | Medium–high | Supporting experiment |
| “Use a bigger embedding model” | Low | Unknown | Low–medium | Baseline/ablation only |
| Replace DP by an end-to-end Video-LLM | Risky | Unknown | Very high | Not recommended for this paper |

The highest-return research question is therefore:

> **Can retrieval improve when the score of a multi-event video path depends not only on whether each frame matches each event, but also on whether consecutive candidate clips realize the entity, state, motion, and temporal transition described between those events?**

That is a paper.

## Highest-value research directions

### Transition-aware structured temporal alignment

This should be the centerpiece.

Instead of representing the query as independent strings

$$
E_1,E_2,\ldots,E_M,
$$

construct structured event nodes:

$$
E_i =
(
\text{entities},
\text{action},
\text{objects},
\text{attributes},
\text{state-before},
\text{state-after},
\text{modality cues}
)
$$

and transition edges:

$$
R_i =
R(E_{i-1},E_i).
$$

For example:

> “A woman enters a kitchen, takes a bottle from the refrigerator, pours water into a glass, then leaves with the glass.”

could become:

$$
E_1: \text{woman enters kitchen}
$$

$$
R_2: \text{same woman; same location; short transition}
$$

$$
E_2: \text{woman takes bottle from refrigerator}
$$

$$
R_3: \text{woman and bottle persist; bottle state changes}
$$

$$
E_3: \text{woman pours water into glass}
$$

$$
R_4: \text{woman and glass persist; movement begins}
$$

$$
E_4: \text{woman leaves carrying glass}.
$$

Now define the path objective as

$$
\boxed{
\mathcal{J}(z_{1:M})
=
\sum_{i=1}^{M} U_i(z_i)
+
\sum_{i=2}^{M} T_i(z_{i-1},z_i)
-
\rho\sum_i\mathbf{1}[z_i=\varnothing]
}
$$

where \(z_i\) can be a candidate frame/clip or a null state.

The critical difference is:

$$
T_i(a,b)
\neq
-\lambda(t_b-t_a).
$$

Instead:

$$
T_i(a,b)
=
\alpha D_i(\Delta t)
+
\beta C_i(a,b)
+
\gamma R_i(a,b)
+
\delta M_i(a,b).
$$

Here:

$$
D_i(\Delta t)
$$

is an **event-conditioned duration potential**;

$$
C_i
$$

measures entity continuity;

$$
R_i
$$

measures semantic state/transition consistency;

$$
M_i
$$

measures motion compatibility.

This changes the interpretation of DP completely.

DP is no longer the contribution.

It is merely the exact/approximate decoder for your structured temporal model.

That is much stronger academically.

### Event-conditioned temporal distance

This is the simplest form of transition modeling and should be implemented first.

Your current system assumes every adjacent event pair receives the same linear gap preference.

Instead learn or predict

$$
P(\Delta t\mid E_{i-1},E_i).
$$

Then:

$$
D_i(\Delta t)
=
\log P(\Delta t\mid E_{i-1},E_i).
$$

A simple initial implementation could use categories:

| Transition type | Example | Expected temporal prior |
|---|---|---|
| atomic continuation | “raises cup → drinks” | very short |
| same-scene sequential | “opens fridge → takes bottle” | short |
| extended action | “starts cooking → serves food” | medium |
| narrative transition | “interview → later outdoor scene” | broad |
| unknown | generic events | weak/uniform |

A stronger implementation predicts parameters from an event-pair embedding:

$$
[\mu_i,\sigma_i]
=
g_\theta(E_{i-1},E_i),
$$

then uses a log-normal-like potential:

$$
D_i(\Delta t)
=
-
\frac{
(\log(1+\Delta t)-\mu_i)^2
}{
2\sigma_i^2
}.
$$

You do not necessarily need large-scale training. A small held-out query set or pseudo-labeling can be enough to test the hypothesis.

Crucially, this immediately eliminates the telescoping problem because

$$
D_i
$$

is different for every transition and is non-linear in \(\Delta t\).

Even before entity tracking or VLM reasoning, this is a meaningful new temporal model.

### Entity-continuity-aware alignment

This is potentially the most impressive extension.

Many difficult TRAKE queries contain an implicit constraint:

> **the same entity participates throughout the event sequence.**

Your current frame score mostly asks whether each event is represented somewhere in the frame.

Suppose:

$$
E_1=\text{man holds umbrella}
$$

and

$$
E_2=\text{man enters car}.
$$

The system can align man A in \(E_1\) to man B in \(E_2\).

A transition-aware model can define:

$$
C_i(a,b)
=
\max_{x\in \mathcal E(a), y\in\mathcal E(b)}
\operatorname{sim}
(\phi(x),\phi(y)),
$$

where \(\mathcal E(a)\) and \(\mathcal E(b)\) are detected entities and \(\phi\) is an appearance embedding.

You already possess important pieces:

- YOLOE detections;
- DINO embedding support;
- frame timestamps;
- raw-video/keyframe infrastructure.

A practical first version does not need sophisticated multi-object tracking across the entire corpus.

Use entity continuity **only inside shortlisted candidate videos**:

$$
\text{retrieval}
\rightarrow
\text{top candidate intervals}
\rightarrow
\text{detect/crop candidate entities}
\rightarrow
\text{appearance continuity score}.
$$

This is computationally much easier.

For general objects, DINO-like crop embeddings should be tested.

For people, start with clothing/body appearance rather than making face recognition a dependency.

The paper claim should not be “we track people.” It should be:

> **persistent-entity evidence is incorporated as a pairwise potential inside multi-event temporal alignment.**

That is more general.

### State-transition-aware retrieval

Entity continuity alone cannot distinguish:

> person holding closed umbrella

from:

> person holding open umbrella.

Or:

> cup empty

from:

> cup filled.

Or:

> person approaching car

from:

> person inside car.

Those are **state transitions**, and they are exactly where independent-frame embeddings become weak.

A transition score can operate on a pair or short sequence of candidate observations:

$$
R_i(a,b)
=
\operatorname{score}
(
\text{visual transition }a\rightarrow b,
\text{text relation }E_{i-1}\rightarrow E_i
).
$$

You have several implementation levels.

The cheapest is structured textual comparison using captions and object states.

A stronger version uses a short local clip encoder.

The strongest but most expensive uses Qwen3-VL on a bounded sequence of frames such as:

$$
\{a-\tau,a,a+\tau,b-\tau,b,b+\tau\}
$$

and asks for a constrained score rather than free-form reasoning.

The prompt should not ask:

> “Does this video match the entire query?”

Instead ask atomic questions such as:

> “Do frames A–C show the same person transitioning from holding the bottle to pouring from it? Return a score from 0 to 1.”

That produces a much more controlled pairwise feature.

Again, do this only for a small candidate lattice.

### Motion-aware local clip refinement

The current keyframe representation is inherently weak for verbs such as:

- approaching;
- overtaking;
- turning;
- falling;
- opening;
- closing;
- throwing;
- receiving;
- exchanging;
- entering;
- leaving.

A static frame may contain excellent object evidence while being ambiguous about the action.

Instead of replacing your current indexing pipeline, use it as Stage A.

Stage A:

$$
\text{keyframe multimodal retrieval}
\rightarrow
\text{high recall}.
$$

Stage B:

$$
\text{candidate timestamps}
\rightarrow
[t-\tau,t+\tau]
$$

from the underlying video.

Stage C:

$$
\text{clip encoder}
\rightarrow
\text{action/motion-aware score}.
$$

Modern temporal-grounding work provides strong justification for using dedicated temporal representations rather than relying exclusively on frozen image embeddings. Sparse-Dense Side-Tuner, for instance, obtains strong grounding results with InternVideo2 while using parameter-efficient adaptation. citeturn15search13 ReVisionLLM and VideoTree further support coarse-to-fine treatment of long videos rather than uniformly processing all frames at high cost. citeturn15search1turn15search8

For your problem, I would test three representations:

$$
\text{single keyframe},
$$

$$
\text{mean/attention pooled adjacent keyframes},
$$

and

$$
\text{true short-video embedding}.
$$

That experiment alone will tell you how much of the remaining error is caused by representation versus alignment.

### Query-conditioned multimodal gating

I consider this the best **low-cost secondary contribution**.

Replace

$$
S_i
=
\frac13S_i^V+
\frac13S_i^C+
\frac13S_i^A
$$

with

$$
S_i
=
\sum_{m}
w_{i,m}S_{i,m}
$$

where

$$
\mathbf w_i
=
\operatorname{softmax}
g(E_i,\mathbf r_i).
$$

\(\mathbf r_i\) can encode evidence availability/reliability:

- event contains quoted speech;
- event contains readable words/numbers;
- event is motion-heavy;
- event is object/scene-heavy;
- ASR candidate score distribution is flat;
- OCR evidence is absent;
- caption confidence is weak;
- modality retrieval entropy is high.

For example:

> “The screen displays 2026.”

should upweight OCR.

> “A reporter says that inflation increased.”

should upweight ASR.

> “The football player kicks the ball.”

should prioritize video.

The first version can even use rule-based gates derived from structured query parsing. A learned gate can follow.

An especially useful confidence statistic is retrieval entropy.

For modality \(m\):

$$
p_{m,j}
=
\frac{\exp(s_{m,j}/\tau)}
{\sum_k \exp(s_{m,k}/\tau)},
$$

and

$$
H_m
=
-\sum_jp_{m,j}\log p_{m,j}.
$$

A sharply peaked modality has potentially useful evidence; a near-uniform modality is likely non-discriminative.

You can condition its contribution on this reliability rather than blindly stretching every modality through row-wise min-max normalization.

### Null-event alignment

I strongly recommend adding this even if it receives only one subsection in the paper.

Current decoding requires

$$
z_i\neq\varnothing
\quad\forall i.
$$

Change the state space to allow

$$
z_i=\varnothing
$$

with penalty \(\rho_i\).

Then an irrelevant or poorly grounded event can be skipped rather than poisoning the entire sequence.

Drop-DTW is strong precedent for the general principle of sequence alignment with outlier dropping. citeturn10search1 StepFormer's order-aware filtering of irrelevant text in multi-step localization reinforces the same motivation in video. citeturn10search2

This is particularly useful because your event segmentation may originate from:

- sentence splitting;
- LLM parsing;
- user-provided TRAKE subqueries.

None of those guarantees every clause corresponds to a unique visible frame.

A neat extension is an **event visibility estimate**:

$$
\rho_i=
h(E_i),
$$

so a clause such as

> “the narrator explains why this happened”

has a lower penalty for being visually skipped than

> “a red truck enters the intersection.”

### Selective VLM verification, not universal VLM reranking

VLM verification is worth using because your code already supports it, but treat it as a **confidence-triggered final stage**.

For example, invoke the VLM only when

$$
\mathcal{J}(P_1)-\mathcal{J}(P_2)<\epsilon
$$

or when the best path has poor entity/transition confidence.

That gives:

$$
\text{cheap retrieval}
\rightarrow
\text{structured DP}
\rightarrow
\begin{cases}
\text{return}, & \text{confident}\\
\text{VLM verify}, & \text{uncertain}.
\end{cases}
$$

This makes computational cost easy to report and avoids framing the paper as another “DP + LVLM” system, a space already represented by Lucifer-TRACE. citeturn14search11

It would also let you make a useful empirical statement:

> selective VLM reasoning delivers most of the verification gain at only \(X\%\) of the VLM calls required by uniform reranking.

That is a much better systems contribution.

## Recommended paper thesis and method

I would structure the actual paper around one central method, provisionally titled:

> **Transition-Aware Multimodal Alignment for Multi-Event Video Retrieval**

I would avoid locking the acronym until checking the final literature for naming collisions.

### Problem formulation

Let the corpus be

$$
\mathcal V=\{V_1,\ldots,V_N\}.
$$

A query describes a temporally ordered event sequence

$$
Q=(E_1,\ldots,E_M).
$$

The target is not merely a video \(V^*\), but an ordered path

$$
P^*=
(z_1^*,\ldots,z_M^*)
$$

such that

$$
t(z_1^*)<\cdots<t(z_M^*)
$$

and the joint path explains the query.

For KIS, the path additionally induces an inferred segment

$$
[\hat s,\hat t]
$$

and representative submission frame.

For TRAKE, expose the full path.

That immediately unifies the two tasks scientifically:

$$
\boxed{
\text{KIS and TRAKE share latent temporal path inference;
they differ mainly in the readout.}
}
$$

That is an elegant framing for the paper.

### Structured event parser

The query parser outputs:

$$
\mathcal G_Q=
(\mathcal E,\mathcal R),
$$

where event \(E_i\) contains:

$$
E_i=
(a_i,o_i,x_i,l_i,s_i,c_i)
$$

for action, object/entities, attributes, location, state, and modality cues.

Edges contain:

$$
R_i=
(\text{continuity},
\text{state change},
\text{temporal relation},
\text{duration class}).
$$

The parser must preserve the original text as a fallback.

This is important: **never let the LLM rewrite become the only representation**.

Store:

$$
E_i^{original},
E_i^{literal},
E_i^{structured},
E_i^{paraphrases}.
$$

Your existing `QueryCandidateSet` is already architecturally close to supporting this.

### Candidate generation

Use the current high-recall multimodal retriever.

For each event:

$$
\mathcal C_i=
\operatorname{TopK}
\{
U_i(f)
\}.
$$

Then construct a video coverage score such as:

$$
A(V)
=
\sum_{i=1}^{M}
\operatorname{LSE}_{f\in V\cap\mathcal C_i}
U_i(f),
$$

or a simpler top-event-score aggregation.

The key criterion should be **event coverage**, not simply the single best frame.

A correct video containing moderately good candidates for all four events is preferable to a video containing one exceptionally good event and no evidence for the others.

Retain top \(B\) videos.

Within each shortlisted video, add temporal neighbors around retrieved anchors.

This forms a much smaller candidate lattice.

### Adaptive unary evidence

For event \(i\) and candidate \(z\):

$$
U_i(z)
=
w_{i,V}S_{i,V}(z)
+
w_{i,C}S_{i,C}(z)
+
w_{i,A}S_{i,A}(z)
+
w_{i,O}S_{i,O}(z)
+
w_{i,B}S_{i,B}(z)
+
w_{i,M}S_{i,M}(z).
$$

Here \(M\) can denote motion/clip evidence.

Weights are query-conditioned and optionally confidence-conditioned:

$$
\mathbf w_i
=
\operatorname{softmax}
g_\theta
(
\phi(E_i),
r_{i,V},\ldots,r_{i,M}
).
$$

The simplest paper version can use a tiny MLP rather than a large trainable model.

### Transition-aware pairwise potential

The main contribution is:

$$
T_i(a,b)=
D_i(a,b)
+
C_i(a,b)
+
R_i(a,b).
$$

A practical decomposition is

$$
T_i(a,b)
=
\lambda_dD_i(\Delta t)
+
\lambda_eC_i(a,b)
+
\lambda_rR_i(a,b).
$$

The full score becomes:

$$
\boxed{
\mathcal{J}(P)=
\sum_i U_i(z_i)
+
\sum_{i=2}^{M}T_i(z_{i-1},z_i)
-
\sum_i\rho_i\mathbf1[z_i=\varnothing].
}
$$

Inference remains monotonic:

$$
t(z_i)>t(z_{i-1})
$$

unless one state is null.

The recurrence is then conceptually:

$$
DP[i,b]
=
U_i(b)+
\max_{a<t_b}
\left[
DP[i-1,a]+T_i(a,b)
\right].
$$

Unlike the current linear-gap recurrence, arbitrary pairwise transitions can no longer use the simple prefix-max trick.

But that is not a problem if this richer DP operates over the shortlisted lattice.

With \(K\) candidates per event:

$$
O(MK^2)
$$

is often perfectly reasonable.

You can further restrict predecessors using a temporal window \(W\):

$$
O(MKW).
$$

This is the key architectural trade:

> **current full-frame DP is extremely cheap because its temporal model is simple; proposed DP becomes richer, so retrieval first reduces its state space.**

That is a coherent algorithmic story.

### Local clip refinement

For every candidate anchor \(z\), generate a local interval:

$$
I(z)=[t_z-\tau_1,t_z+\tau_2].
$$

Encode it with a video backbone only after coarse retrieval.

Then obtain:

$$
S_{motion}(E_i,z)
=
\cos(
\phi_{text}(E_i),
\phi_{video}(I(z))
).
$$

For pairwise transitions, optionally encode:

$$
I(a,b)
$$

or bounded samples spanning the transition.

One important experiment is to determine how much temporal context is needed:

$$
\tau\in
\{0,1s,2s,4s,8s\}.
$$

That produces a useful plot:

$$
\text{accuracy vs temporal window vs latency}.
$$

This is the kind of result reviewers remember.

### KIS readout

Do not automatically use the upper-middle aligned event.

After selecting a path, infer

$$
[\hat s,\hat t]
=
[t(z_1),t(z_M)]
$$

or use event-dependent start/end margins.

Then choose a representative KIS frame using an explicit criterion:

$$
f^*
=
\arg\max_{f\in[\hat s,\hat t]}
\left[
\sum_i\alpha_iU_i(f)
+
\eta\,\text{distinctiveness}(f)
\right].
$$

A simpler baseline is the frame associated with the most discriminative event:

$$
i^*
=
\arg\max_i
\left(
U_i(z_i)-U_i^{(2nd)}
\right),
$$

then submit \(z_{i^*}\).

Test this against the current upper-middle policy.

It may be a surprisingly easy KIS improvement.

### TRAKE readout

TRAKE naturally returns:

$$
(z_1,\ldots,z_M).
$$

But the system should also retain:

- unary score per event;
- transition score per edge;
- modality weights;
- null states;
- confidence.

That gives you excellent qualitative visualizations for the paper.

Instead of showing:

> “our result is correct,”

show why:

$$
\begin{array}{c|c|c|c}
\text{Event} & \text{Frame} & U_i & T_i\\
\hline
E_1 & 01{:}22 & .81 & -\\
E_2 & 01{:}27 & .74 & .88\\
E_3 & 01{:}31 & .79 & .91
\end{array}
$$

and compare with baseline DP choosing semantically stronger but transition-inconsistent frames.

### The paper's hypotheses

I would freeze the research around a small set of falsifiable hypotheses.

**H1 — Transition hypothesis**

$$
\text{Unary + event-conditioned pairwise transitions}
>
\text{Unary + linear-gap DP}
$$

particularly for queries containing action sequences and state changes.

**H2 — Modality hypothesis**

$$
\text{query-conditioned fusion}
>
\text{fixed global weights}
$$

particularly when only one modality carries decisive evidence.

**H3 — Motion hypothesis**

$$
\text{local clip refinement}
>
\text{keyframe-only retrieval}
$$

particularly for time-sensitive verbs.

**H4 — Robust alignment hypothesis**

$$
\text{null-aware DP}
>
\text{mandatory-event DP}
$$

under syntactic over-segmentation, query expansion noise, and non-visual clauses.

**H5 — Temporal-reasoning hypothesis**

The relative improvement of the proposed system should be larger on **wrong-order hard negatives** than on ordinary negatives.

This last hypothesis is particularly valuable because it tests the claimed mechanism rather than merely leaderboard accuracy.

### What should count as the core contribution

I would keep the contribution statement disciplined.

The paper should say approximately:

> We first identify a limitation of conventional linear-gap multi-event DP: under monotonic alignment, the cumulative linear gap penalty collapses to an overall segment-span prior and cannot express event-specific transitions.

> We introduce a structured temporal retrieval formulation combining multimodal unary evidence with event-conditioned pairwise transition potentials and optional null-event states.

> We implement the formulation in a coarse-to-fine candidate lattice that enables motion/entity/state reasoning without exhaustive expensive video processing.

> We evaluate not only standard retrieval accuracy but also order-reversal, entity-consistency, and modality-corruption stress tests to determine whether improvements arise from temporal reasoning.

That is a much more defensible research contribution than introducing five loosely connected competition tricks.

## Experimental program and ablations

The experimental design will determine whether this becomes a paper or merely an improved system.

### Build a frozen research benchmark before tuning the new method

The repository itself correctly notes that an HCMAI accuracy improvement cannot be established without a frozen evaluation set.

This should be your first scientific deliverable.

For each research query, store:

$$
Q,\quad
V^*,\quad
[s^*,t^*],
$$

and ideally event-level grounding:

$$
(E_i,[s_i^*,t_i^*]).
$$

For TRAKE-style queries, event-level annotations are extremely valuable.

A minimum viable research set would be on the order of a few hundred carefully controlled event instances rather than thousands of weak labels. The precise size should follow annotation capacity; quality is more important than pretending the competition corpus itself provides supervision that it does not.

For any learned calibration/gating component, split **by video**, not by frame, to prevent closely related frames from the same broadcast leaking across train and test.

Do not continuously modify the test annotations while developing.

Create:

$$
D_{train},D_{val},D_{test}
$$

once, hash the manifest, and report the hash/run identifier.

### Separate video retrieval from temporal alignment

A major evaluation mistake would be to report only final Recall@K.

That makes it impossible to tell whether the paper improves semantic retrieval or temporal reasoning.

Report at least three layers.

#### Video retrieval

Measure:

$$
R@1,\ R@5,\ R@10,\ R@20
$$

and MRR.

This answers:

> Did the method locate the correct video?

#### Event grounding

For aligned timestamp \(\hat t_i\), define:

$$
Hit_\delta(i)=
\mathbf1[
d(\hat t_i,[s_i,t_i])\leq \delta
].
$$

Report several tolerances such as:

$$
\delta\in\{1s,2s,5s\},
$$

adapted to your actual keyframe density.

Also report median/mean temporal error.

#### Path-level success

This is particularly important for TRAKE.

Define:

$$
AllHit@\delta
=
\frac1{|Q|}
\sum_q
\mathbf1[
\forall i,\ Hit_\delta(q,i)=1
].
$$

A four-event query where three events are correct and one is wrong should not look equivalent to four independent single-event successes.

Also report average per-event hit rate so the metric is not excessively harsh.

### Evaluate intervals as intervals

Since the user's query describes a short temporal segment \([s,t]\), report temporal IoU whenever interval ground truth exists:

$$
tIoU
=
\frac{
|[\hat s,\hat t]\cap[s,t]|
}{
|[\hat s,\hat t]\cup[s,t]|
}.
$$

This will be particularly useful for KIS because it separates:

> correct video but poor localization

from

> correct and temporally precise retrieval.

### Create temporal hard-negative subsets

This is where the proposed paper can become much stronger than a challenge report.

Construct subsets where ordinary semantics are deliberately insufficient.

**Wrong-order cases**

Both videos contain A, B, C, but:

$$
V^+:A\rightarrow B\rightarrow C
$$

while

$$
V^-:A\rightarrow C\rightarrow B.
$$

ArrowGEV's emphasis on time-sensitive versus time-insensitive events supports the importance of explicitly testing temporal direction rather than only generic semantic similarity. citeturn15search15

**Entity-switch cases**

Correct:

$$
\text{person A performs }E_1,E_2,E_3.
$$

Hard negative:

$$
A:E_1,\quad B:E_2,\quad A:E_3.
$$

**State-transition cases**

Correct:

$$
\text{closed}\rightarrow\text{opening}\rightarrow\text{open}.
$$

Hard negative contains all three states but in the wrong sequence or for different objects.

**Motion/static-confusion cases**

Examples:

$$
\text{standing near car}
\quad\text{vs}\quad
\text{entering car},
$$

$$
\text{holding ball}
\quad\text{vs}\quad
\text{throwing ball}.
$$

**Modality-conflict cases**

Visual evidence indicates one event while ASR/OCR contains lexical distractors.

These subsets allow you to say:

> the gain is specifically concentrated where temporal reasoning is necessary.

That is significantly stronger than:

> our overall Recall@5 increased by 3%.

### Test query corruption explicitly

Create controlled perturbations:

$$
Q \rightarrow Q_{\text{split-noise}}
$$

by adding an unnecessary event;

$$
Q \rightarrow Q_{\text{missing}}
$$

by dropping one event;

$$
Q \rightarrow Q_{\text{paraphrase}}
$$

with alternate language;

$$
Q \rightarrow Q_{\text{abstract}}
$$

with a non-visible clause.

Then compare mandatory alignment against null-aware alignment.

This is where Drop-DTW-inspired robustness should become visible. citeturn10search1

### The essential ablation table

The paper should contain something structurally like:

| Model | Adaptive modality | Transition duration | Clip motion | Entity continuity | Null event | VLM verify | TRAKE AllHit | KIS R@1 |
|---|---|---|---|---|---|---|---:|---:|
| Current baseline | ✗ | linear | ✗ | ✗ | ✗ | ✗ | … | … |
| + calibrated unary | ✓ | linear | ✗ | ✗ | ✗ | ✗ | … | … |
| + event duration | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | … | … |
| + motion transition | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | … | … |
| + null states | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | … | … |
| + entity continuity | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | … | … |
| Full + selective verify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | … | … |

Do not skip the first row.

Your **existing code is the most important baseline** because reviewers need to see that temporal transition modeling—not simply a completely different system—is responsible for the gain.

Also include:

$$
\lambda_{gap}=0
$$

as a baseline.

That will reveal how much the current linear gap term is actually contributing.

### Explicitly test the telescoping hypothesis

This should be a dedicated experiment because it supports the theoretical motivation.

Construct synthetic score matrices where unary scores are fixed but intermediate pacing differs.

For example, with three events:

$$
P_A=(0,2,10)
$$

and

$$
P_B=(0,8,10).
$$

Give both paths identical unary scores.

Your current objective must assign identical temporal cost.

Then construct a query where the expected transition is:

$$
E_1\rightarrow E_2
$$

rapidly, followed by a long delay before \(E_3\).

The proposed event-conditioned transition model should prefer \(P_A\).

A small theoretical proposition plus this controlled experiment could be one of the cleanest parts of the paper.

### Compare event segmentation strategies

Evaluate:

$$
\text{sentence split},
$$

$$
\text{LLM event split},
$$

and

$$
\text{LLM structured event graph}.
$$

But do not make this an enormous prompt-engineering exercise.

Measure:

- average number of events;
- event-level grounding recall;
- downstream path accuracy;
- null-event usage;
- latency.

A structured parser is useful only if it improves retrieval.

### Compare fixed and adaptive fusion properly

At minimum:

$$
V,
C,
A,
V+C,
V+A,
C+A,
V+C+A,
$$

then:

$$
\text{fixed optimized weights}
$$

versus

$$
\text{query-conditioned weights}.
$$

This distinction matters. Otherwise reviewers can reasonably argue that the adaptive gate only wins because the default \(1/3,1/3,1/3\) weights were suboptimal.

First tune the best global fixed weights on validation.

Then beat **that**, not only the default.

Similarly, compare:

$$
\text{row min-max}
$$

against alternative calibration approaches before claiming the gate itself is responsible.

### Compare keyframes against clips only where it matters

Do not simply report an overall clip-model gain.

Break queries into:

$$
Q_{static}
$$

and

$$
Q_{motion}.
$$

If your hypothesis is correct, then:

$$
\Delta_{motion}
\gg
\Delta_{static}.
$$

That mechanism-based result is much more compelling.

### Efficiency needs to be a first-class result

The challenge's 2026 framing includes automated intelligent retrieval, making runtime increasingly relevant. citeturn11search3

Report:

$$
P50,\quad P95
$$

for:

- query parsing;
- global retrieval;
- candidate construction;
- DP;
- clip refinement;
- VLM verification;
- total request.

Also report:

- number of candidate videos;
- candidates per event;
- number/percentage of queries invoking VLM verification;
- GPU memory;
- number of expensive model calls.

Plot:

$$
\text{accuracy}
\quad\text{vs}\quad
K
$$

for candidate lattice size.

This establishes whether pairwise reasoning is practically useful rather than theoretically attractive.

### Use a public benchmark for one external-validity experiment

Your main benchmark should remain challenge-aligned because that is the paper's real problem.

But one external experiment would substantially strengthen the work.

The closest families are multi-sentence/paragraph temporal grounding and multi-step localization rather than generic text-video retrieval. Video Paragraph Grounding explicitly studies localization of multiple sentences whose semantic relationships and temporal order matter, which is structurally much closer to TRAKE than single-moment retrieval. StepFormer likewise provides a useful multi-step localization setting. citeturn10search2

You do not need to chase SOTA on every public benchmark.

A stronger claim is:

> the same transition-aware decoder improves a standard temporal-grounding backbone on an external ordered-event dataset.

That proves the algorithm is not merely a hand-crafted HCMAI heuristic.

## Paper positioning, risks, and open questions

### The paper I would write

A concise title could be:

> **Transition-Aware Multimodal Alignment for Multi-Event Video Retrieval**

The abstract should begin from the failure of existing retrieval:

> Multi-event video queries describe not only what appears in individual moments but also how entities, actions and states evolve across time.

Then identify the problem:

> Existing frame-retrieval pipelines followed by monotonic dynamic programming primarily combine independent event-frame compatibility with chronological constraints.

Then your theoretical observation:

> For the commonly used linear temporal-gap objective under strict monotonic alignment, pairwise gap costs collapse into a penalty on total path span and therefore cannot represent event-specific temporal transitions.

Then method:

> We introduce a transition-aware multimodal alignment objective combining adaptive unary evidence, event-conditioned pairwise temporal potentials, and optional null-event states, decoded efficiently over a coarse-to-fine candidate lattice.

Then evaluation:

> On HCMAI multi-event retrieval and targeted wrong-order/entity/state stress sets, the proposed formulation improves both video retrieval and complete-path localization, particularly on queries requiring genuine temporal reasoning.

Obviously, the final performance claim must wait for experiments.

### The paper should not be positioned as a system paper full of unrelated modules

Avoid the contribution list:

> We use SigLIP2, BGE-M3, Qwen, BM25, ASR, OCR, YOLOE, DP, DINO, reranking, query expansion, etc.

That describes software architecture, not scientific novelty.

Instead:

$$
\boxed{\text{one problem}
\rightarrow
\text{one central hypothesis}
\rightarrow
\text{one structured model}
\rightarrow
\text{controlled experiments}}
$$

The system components support that hypothesis.

### What I would explicitly avoid as the main direction

I would **not** lead with “improved dynamic programming.” DANTE already occupies that space. citeturn12academia1

I would **not** lead with “DP + LVLM verification.” Lucifer-TRACE already has that positioning. citeturn14search11

I would **not** lead with “query augmentation + web image search.” QUEST, MADTempo, and RAPID already make closely related contributions. citeturn12academia1turn11academia17turn12academia3

I would **not** lead with “temporally consistent captioning.” U-CESE already introduces ReCap. citeturn13academia24

I would **not** lead with “adaptive keyframe extraction.” Both U-CESE and another HCMC 2025 system already contain keyframe-selection contributions. citeturn13academia24turn12academia0

I would **not** immediately replace the entire architecture with an end-to-end Video-LLM. Recent literature shows powerful Video-LLM grounding models, but it also shows considerable architectural effort specifically to make long-video temporal reasoning tractable—hierarchical search, specialized temporal experts, temporal training data, token reduction, and dedicated temporal representations are recurring solutions. citeturn15search1turn15search2turn15search3turn15search4 Your retrieval-first architecture is therefore not obsolete; it is a strong foundation for structured search.

### A realistic research sequence

The highest-value implementation order is:

$$
\boxed{
\text{freeze benchmark}
\rightarrow
\text{analyze current errors}
\rightarrow
\text{candidate lattice}
\rightarrow
\text{event-conditioned gap}
\rightarrow
\text{null states}
\rightarrow
\text{clip motion}
\rightarrow
\text{adaptive fusion}
\rightarrow
\text{entity/state transition}
\rightarrow
\text{selective VLM verification}
}
$$

The key point is that you should **not implement everything before measuring anything**.

The first decisive experiment is very small:

$$
\text{current DP}
\quad\text{vs}\quad
\text{event-conditioned transition DP}.
$$

If that does not improve path-level accuracy on temporally difficult queries, investigate why before introducing more components.

The next decisive experiment is:

$$
\text{keyframe unary}
\quad\text{vs}\quad
\text{local clip unary}.
$$

That tells you whether motion representation is the dominant bottleneck.

Then:

$$
\text{fixed fusion}
\quad\text{vs}\quad
\text{adaptive fusion}.
$$

Only afterwards should entity/state continuity and VLM verification be added.

### The most valuable qualitative figure

One figure should show a query and two candidate paths.

For example:

**Query**

> A man picks up a box → carries it toward a truck → places the same box inside the truck.

**Current DP**

$$
f_{21}:
\text{man A holding box}
$$

$$
\downarrow
$$

$$
f_{48}:
\text{man B walking beside truck}
$$

$$
\downarrow
$$

$$
f_{61}:
\text{box inside truck}.
$$

High unary scores, correct chronological order, **wrong event chain**.

**Proposed**

$$
f_{24}
\rightarrow
clip_{31}
\rightarrow
f_{43}
$$

with:

$$
\text{same-person consistency}=0.91,
$$

$$
\text{box continuity}=0.87,
$$

$$
\text{transition score}=0.89.
$$

That single figure communicates the entire paper more effectively than an architecture diagram full of models.

### The strongest theoretical statement available from the current code

I would seriously consider formalizing the following proposition.

**Proposition.** Under strictly increasing timestamps, a constant linear adjacent-gap penalty

$$
G(P)
=
-\lambda
\sum_{i=2}^{M}
(t_i-t_{i-1})
$$

is equivalent to

$$
G(P)
=
-\lambda(t_M-t_1).
$$

Therefore, conditional on the first and last selected timestamps, the gap term is invariant to all intermediate event timestamps.

**Consequence.**

The objective cannot prefer one internal temporal arrangement over another based on event-specific pacing; all discrimination among such paths comes from unary event-frame scores and other constraints.

This observation is elementary mathematically, which is actually a strength: it is easy for reviewers to verify, directly applies to your existing baseline, and precisely motivates the proposed pairwise model.

Do not overclaim that this is a novel theorem.

The research novelty is:

> recognizing its practical implication for multi-event video retrieval and replacing the degenerate temporal prior with semantically conditioned transition potentials.

### What success would look like scientifically

The ideal result is not merely:

$$
R@1:+X.
$$

It is a pattern such as:

$$
\Delta_{\text{ordinary queries}}
=
+2.1
$$

but

$$
\Delta_{\text{wrong-order}}
=
+11.8,
$$

$$
\Delta_{\text{state-change}}
=
+13.4,
$$

$$
\Delta_{\text{entity-continuity}}
=
+9.7.
$$

Those numbers are illustrative, not predictions.

Such a pattern would demonstrate that the new method improves exactly the class of queries predicted by the theory.

Likewise, for multimodal gating, a convincing result would be:

$$
\Delta_{\text{speech queries}}
\gg 0
$$

when adaptive gating increases ASR weight,

while:

$$
\Delta_{\text{motion queries}}
\gg 0
$$

when it increases video/clip evidence.

Mechanistic evaluation is much more persuasive than aggregate leaderboard optimization.

### Open questions and current limitations

The ZIP contains the research/runtime implementation but not a complete frozen benchmark with query-level ground truth and existing experimental result tables. Consequently, I cannot yet determine empirically whether the current dominant failure mode is candidate recall, temporal alignment, keyframe sparsity, modality fusion, or event parsing. The roadmap above is therefore based on architectural analysis plus the current literature, not on measured error frequencies from your own test set.

The user-stated corpus size of **873 videos** was treated as the authoritative size for this project; I did not find a current official public source establishing that exact corpus count, and the challenge's public pages describe the broader multimedia-retrieval task rather than the precise dataset snapshot in your repository. citeturn11search3

The exact special-session publication mechanics also need to be treated carefully. The AI Challenge HCMC 2026 site states that selected methods may be invited to the SoICT 2026 special session and says those proceedings are published by ACM, whereas the current general SoICT 2026 website describes Springer CCIS proceedings and a 12-page main-paper format. citeturn11search3turn11search2 This administrative inconsistency does not affect the scientific recommendation, but the final manuscript template should follow the instructions specifically sent through the competition route rather than assumptions from the normal SoICT track.

Finally, the most recent challenge literature makes the novelty boundary unusually clear. DANTE already establishes DP for TRAKE; Lucifer-TRACE already establishes DP plus LVLM verification; U-CESE already establishes clip-based retrieval plus temporally consistent captions; MADTempo/QUEST/RAPID already occupy query augmentation/OOD retrieval territory. citeturn12academia1turn14search11turn13academia24turn11academia17turn12academia3

That leaves your strongest defensible research territory as:

$$
\boxed{
\begin{aligned}
&\textbf{independent event-frame matching}\\
&\qquad\Downarrow\\
&\textbf{structured event-transition retrieval}
\end{aligned}
}
$$

or, more concretely,

$$
\boxed{
\text{What happens at each moment}
+
\text{how one moment transforms into the next}
}
$$

rather than merely

$$
\boxed{
\text{what happens at each moment}
+
\text{timestamps must increase}.
}
$$

That is the conceptual step I would build the SoICT paper around.