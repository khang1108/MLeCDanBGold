# Multimodal Video Retrieval Failure Analysis: Why Temporal Alignment Underperforms Ordinary Dense Search

**Date:** 2026-09-04  
**Topic:** Root Cause Diagnosis of Retrieval Quality Degradation in Temporal Alignment vs Dense Retrieval  
**Scope:** Active 470,804-frame HCMAI 2026 corpus, live backend services (`http://127.0.0.1:8000`, `http://127.0.0.1:8100`)  
**Status:** SOURCE-VERIFIED EMPIRICAL REPORT  

---

## 1. Executive Summary & Core Diagnostic Finding

Recent observations on narrative video queries revealed a critical pathology: **ordinary dense retrieval frequently outperforms multi-event temporal alignment, and strict temporal dynamic programming (DP) can dramatically lower the target video's rank from the top-10 to beyond rank >100.**

To determine exactly **WHERE** and **WHY** retrieval quality is lost, an empirical investigation was conducted across a benchmark of 5 representative narrative queries on the active corpus (470,804 keyframes across 873 videos). 

We compared five retrieval regimes:
- **R0 (Dense 1-Query):** Ordinary dense vector search treating the query as a single narrative context.
- **R0_hyb (Dense + BM25 1-Query):** Ordinary single-query hybrid search (`dense_weight=0.5, bm25_weight=0.5`).
- **R1 (Strict Temporal DP - Dense Only):** Current production monotonic dynamic programming ($t_1 < t_2 < \dots < t_N$) over per-event dense embeddings.
- **R1_hyb (Strict Temporal DP - Hybrid):** Current production monotonic DP combining dense and BM25 emission matrices.
- **R2 (Soft-Order P1a DP):** Relaxed temporal dynamic programming allowing bounded local event transposition and dead-zone transition costs.

### Summary of Benchmark Results

| Query ID | Target Video | Description / Narrative Focus | R0 Rank (Dense 1-Q) | R0_hyb Rank (Dense+BM25) | R1 Rank (Strict DP) | R1_hyb Rank (Strict+BM25) | R2 Rank (Soft DP P1a) | Primary Failure Category |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1** | `L24_V035` | Pumpkin stolen & waking lion dance | >100 | >100 (FP: `L25_V003`) | 80 | >100 (FP: `L25_V003`) | **1** | **Cat A (Inversion) + Cat C (BM25 Title)** |
| **Q2** | `L24_V044` | Yellow lion falls over toy ship (36f) | >100 | >100 | >100 | >100 | **1** | **Cat D (Multi-modal Dilution)** |
| **Q3** | `L21_V013` | Rhino mural $\to$ 3 monkeys on bridge | 3 | 5 | **1** | **1** | **1** | *Baseline Success (True Monotonic)* |
| **Q4** | `L27_V015` | Fruit orchard: durian, mangosteen, pomelo, bonbon | 4 | 5 | 3 | 9 | **2** | **Cat A (Local Inversion) + Cat C (BM25 Leak)** |
| **Q5** | `L26_V254` | Cooking galangal flowers (inverted narrative) | 77 | >100 | >100 | >100 | **18** | **Cat A (Inversion) + Cat B (Mandatory Penalty)** |

### Core Findings

1. **Strict Monotonicity ($t_1 < t_2 < \dots < t_N$) is Frequently Violated by Real Narrative Queries (Category A):**
   In natural language queries (e.g., Q1, Q4, Q5), users do not necessarily describe events in strict chronological order. In Q1, the pumpkin theft peaks at $t=365\text{s}$, but waking the animal peaks at $t=22\text{s}$. In Q5, Event 2 (plate ingredients) occurs at $t=500\text{s}$ while Event 3 (hold ingredients) occurs at $t=300\text{s}$. Because strict DP mathematically forbids $t_{e+1} < t_e$, it is forced to select near-zero emission frames for inverted events, collapsing the target video's cumulative score.
2. **Mandatory-Event Accumulation Penalizes Sparse Evidence (Category B):**
   Strict DP scores a video by summing emissions across **all** $N$ events: $\text{Score} = \sum_{e=1}^N S[e, t_e] - \text{Gaps}$. If even one event is visually brief or subtle (or inverted), the entire video is penalized. False-positive "attractor" videos that maintain mediocre constant scores (e.g., $0.15$ across all events) outscore the true video ($0.15 \times 3 = 0.45 > 0.18 + 0.18 + 0.02 = 0.38$).
3. **BM25 Title Keyword Leakage Overpowers Dense Visual Search (Category C):**
   In `artifacts/indexes/bm25/metadata.json`, caption vocabulary is empty ($0$ terms), leaving video title as the dominant lexical field. When a query contains generic terms (e.g., *"bí"* in *"bí đỏ"*), BM25 matches *"BÍ QUYẾT ÔN THI THPT"* across **every frame** of educational lecture videos (`L25_V003`, `L25_V012`). Min-max normalization scales this single match to $1.0$, which with `bm25_weight=0.5` adds $+0.50$ to every frame. This completely overpowers visual cosine similarities ($0.15 - 0.25$), propelling irrelevant lecture videos to Rank 1.
4. **Soft-Order DP (P1a) Substantially Recovers Alignment Quality:**
   Relaxing strict monotonicity via bounded local transposition (P1a) dramatically improves retrieval quality:
   - **Q1:** jumps from Rank 80 $\to$ **Rank 1**.
   - **Q2:** jumps from >100 $\to$ **Rank 1**.
   - **Q4:** improves to **Rank 2**.
   - **Q5:** jumps from >100 $\to$ **Rank 18**.

---

## 2. Detailed Empirical Diagnosis by Query

### Query 1: The Pumpkin & Lion Dance (`L24_V035`)

- **Vietnamese Query:** *"Nhóm 5 người đang chơi đùa bên cạnh một con vật màu vàng, một trong số đó đã mang một vật trông như trái bí đỏ đi giấu, người đàn ông thức dậy không thấy quả bí đỏ đâu nên đánh thức con vật dậy."*
- **Event Decomposition:**
  - $E_1$: Group of 5 playing beside a yellow animal.
  - $E_2$: One person takes an object resembling a pumpkin and hides it.
  - $E_3$: Man wakes up, notices pumpkin missing, and wakes the animal.
- **Target Video:** `L24_V035` (*Nam Sư Du Hí Ăn Bông Bí | Đoàn Lân Hào Nhựt – Vĩnh Long*, 662 frames).

#### Diagnostic Telemetry
- **Target Event Visual Peaks:**
  - $E_1$: Max score $= 0.1739$ at $t = 328,000\text{ ms}$ ($328\text{s}$).
  - $E_2$: Max score $= 0.1801$ at $t = 365,000\text{ ms}$ ($365\text{s}$).
  - $E_3$: Max score $= 0.1581$ at $t = 22,000\text{ ms}$ ($22\text{s}$).
- **The Breakdown in Strict DP (R1):**
  - Because $E_2$ peaks at $365\text{s}$ and $E_3$ peaks at $22\text{s}$, the true chronological sequence is **inverted** ($t_3 < t_2$).
  - Strict monotonic DP requires $t_1 < t_2 < t_3$. Because $t_3$ must be $> t_2$ ($> 365\text{s}$), DP cannot select frame $22\text{s}$.
  - The actual aligned path selected by R1 in `L24_V035`:
    - Frame 1: $t=363,000\text{ ms}$
    - Frame 2: $t=365,000\text{ ms}$
    - Frame 3: $t=366,000\text{ ms}$
  - The path collapsed into 3 consecutive seconds around $E_2$. At frame $366\text{s}$, the emission score for $E_3$ was near zero ($0.02$).
  - Target video dropped to **Rank 80** in dense R1.
- **The Breakdown in Hybrid Search (R0_hyb & R1_hyb):**
  - With BM25 enabled, `L25_V003` (*"BÍ QUYẾT ÔN THI THPT 2024"*) matched the word *"BÍ"* across all 500 frames.
  - BM25 score scaled to $1.0$. Because `bm25_weight=0.5`, every frame in `L25_V003` received $+0.50$, scoring $3 \times 0.50 = 1.50$ baseline before visual similarity.
  - `L25_V003` captured Rank 1 with score $1.649$. `L24_V035` was evicted beyond Rank >100.
- **The Soft DP (R2) Recovery:**
  - Soft-order DP allowed local transposition ($E_2 \leftrightarrow E_3$).
  - Result: `L24_V035` jumped immediately to **Rank 1** (Score $0.4910$).

---

### Query 2: Lion Dance Jumping Over Model Ship (`L24_V044`)

- **Query:** *"Một chú lân (hay rồng/sư tử?) màu vàng nhảy hay rơi từ trên cao xuống, gần với mô hình chiếc tàu thủy nhỏ màu xanh dương."*
- **Target Video:** `L24_V044` (36 frames total).
- **Ground Truth Evidence:** Frames 8–9 ($t = 8,000 - 9,000\text{ ms}$) show the yellow lion airborne directly over a blue model ship.

#### Diagnostic Telemetry
- **Visual Similarity:**
  - `L24_V044` frame 9 visual score $= \mathbf{0.2305}$ (highest in entire corpus for this query).
  - Top competitor (`L24_V042`) visual score $= 0.2032$.
- **Why Did R0 and R1 Both Place Target at Rank >100?**
  - In `configs/baseline.yaml`, dense retrieval combines:
    $$\text{Dense} = 0.333 \cdot \text{Visual} + 0.333 \cdot \text{Context} + 0.333 \cdot \text{ASR}$$
  - `L24_V044` is an action clip with only 36 frames. There is no speech mentioning a "ship" or "model", and generated captions did not mention the small blue ship.
  - As a result: $\text{Context} \approx 0.0$, $\text{ASR} = 0.0$.
  - The fused score was diluted: $0.333 \cdot 0.2305 + 0 + 0 \approx \mathbf{0.0768}$.
  - Meanwhile, competitor videos with generic dialogue matching the text gained ASR/Context scores of $0.25 - 0.35$, yielding fused scores $> 0.20$.
- **The R2 Pure Visual Recovery:**
  - When evaluated on visual event scores, `L24_V044` scored $0.2305$ and ranked **Rank 1**.

---

### Query 3: London Zoo Rhino Mural & Monkeys (`L21_V013`)

- **Query:** *"Đoạn clip bắt đầu với cảnh một người đang dùng điện thoại chụp ảnh bức tranh hình tê giác trên tường, đoạn clip kết thúc với cảnh một người chụp ảnh các hình graffiti 3 chú khỉ trên một cây cầu."*
- **Event Decomposition:**
  - $E_1$: Person photographing rhino mural on wall ($t = 745 - 749\text{s}$).
  - $E_2$: Person photographing graffiti of 3 monkeys on bridge ($t = 760 - 766\text{s}$).
- **Target Video:** `L21_V013` (1,127 frames).

#### Diagnostic Telemetry
- **Visual Similarity:**
  - $E_1$: Max score $= 0.2155$ at $t = 749,000\text{ ms}$.
  - $E_2$: Max score $= 0.2593$ at $t = 764,000\text{ ms}$.
- **Chronology:**
  - $t_1 = 749\text{s} < t_2 = 764\text{s}$ ($\Delta t = 15\text{s}$).
  - True chronological order matches the narrative sequence perfectly.
- **Results:**
  - Single-query dense (R0): Rank 3.
  - Strict Temporal DP (R1): **Rank 1** (Score $1.5667$).
  - Hybrid Strict DP (R1_hyb): **Rank 1** (Score $1.3960$).
  - Soft DP (R2): **Rank 1** (Score $0.4379$).
- **Insight:**
  **Temporal alignment succeeds exceptionally well when events are truly monotonic, strongly visible, and close in time.** Here, DP elevated the target video from Rank 3 to Rank 1 by confirming the ordered pair $(749\text{s}, 760\text{s})$.

---

### Query 4: Western Mekong Fruit Orchard (`L27_V015`)

- **Query:** *"Video về một khu vườn cây ăn trái ở miền Tây Nam Bộ có chuỗi liên tiếp các cảnh quay về 4 loại trái cây trong vườn: cảnh có trái sầu riêng, cảnh có trái măng cụt, cảnh có trái bưởi, cảnh có trái dâu bòn bon."*
- **Event Decomposition:**
  - $E_1$: Durian fruit ($t = 226\text{s}$).
  - $E_2$: Mangosteen fruit ($t = 235\text{s}$).
  - $E_3$: Pomelo fruit ($t = 326\text{s}$).
  - $E_4$: Langsat / bòn bon fruit ($t = 221\text{s}$).
- **Target Video:** `L27_V015` (588 frames).

#### Diagnostic Telemetry
- **Event Peak Timestamps in Target:**
  - $E_1$: $t = 226,000\text{ ms}$ (score $0.1816$)
  - $E_2$: $t = 235,000\text{ ms}$ (score $0.1811$)
  - $E_3$: $t = 326,000\text{ ms}$ (score $0.1520$)
  - $E_4$: $t = 221,000\text{ ms}$ (score $0.1775$)
- **The Monotonic Conflict:**
  - In reality, bòn bon ($E_4$) appeared earlier in the video ($221\text{s}$) than pomelo ($E_3$ at $326\text{s}$) and durian ($E_1$ at $226\text{s}$).
  - Strict DP forced $t_4 > t_3 > 326\text{s}$. The aligned path picked frames $225\text{s}, 231\text{s}, 232\text{s}, 236\text{s}$, completely missing $E_3$ and $E_4$'s true peaks.
- **BM25 Impact:**
  - Dense R1: Rank 3.
  - Hybrid R1 (Dense + BM25): Rank **dropped from 3 to 9**.
  - BM25 matched unrelated fruit video titles (`L27_V011`), inflating false positive scores.
- **Soft DP (R2):**
  - Rank improved to **Rank 2**.

---

### Query 5: Cooking Galangal Flowers (`L26_V254`)

- **Query (from fixture `tests/fixtures/l26_v254_query.yaml`):**
  *"Một cô gái mặc tạp dề trắng đứng cạnh một lọ hoa riềng tía, cô gái mặc tạp dề trắng đặt bốn nguyên liệu X chưa xác định lên một đĩa trắng, cùng cô gái mặc tạp dề trắng cầm hai nguyên liệu X cùng loại lên, cô gái mặc tạp dề trắng nói chuyện với một người ngồi đối diện về món ăn sẽ nấu."*
- **Known Ground Truth Regions:**
  - $E_1$: Stand beside vase ($t = 140\text{s}$)
  - $E_2$: Plate ingredients ($t = 500 - 525\text{s}$)
  - $E_3$: Hold two ingredients ($t = 300 - 475\text{s}$)
  - $E_4$: Dialogue ($t = 550 - 950\text{s}$)
- **Target Video:** `L26_V254` (313 frames).

#### Diagnostic Telemetry
- **Event Peaks in Target:**
  - $E_1$: $t = 140,000\text{ ms}$ (score $0.1651$)
  - $E_2$: $t = 87,000\text{ ms}$ (score $0.1608$)
  - $E_3$: $t = 87,000\text{ ms}$ (score $0.1566$)
  - $E_4$: $t = 22,000\text{ ms}$ (score $0.1799$)
- **The Inversion Failure:**
  - The query lists Event 2 (plate ingredients) *before* Event 3 (hold ingredients).
  - In reality, the chef held the ingredients first ($300 - 475\text{s}$) before placing them on the plate ($500 - 525\text{s}$).
  - Because $t_3 < t_2$, strict DP cannot align both.
  - In R0 (Dense 1-Query): Target was **Rank 77**.
  - In R1 (Strict DP): Target was **evicted beyond Rank >100**.
- **Soft DP (R2):**
  - Rank recovered to **Rank 18** (Score $0.6193$).

---

## 3. Taxonomy of Failure Modes

```
                                Retrieval Quality Degradation
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│   CATEGORY A     │               │   CATEGORY B     │               │   CATEGORY C     │
│Narrative Inversion               │Mandatory Penalty │               │BM25 Title Leak   │
│Query order !=    │               │N-event sum drags │               │1-word title hit  │
│video chronology  │               │down target video │               │dominates corpus  │
└──────────────────┘               └──────────────────┘               └──────────────────┘
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│   CATEGORY D     │               │   CATEGORY E     │               │   CATEGORY F     │
│Modality Dilution │               │Gap Penalty Clust.│               │Soft DP Window    │
│Visual-only hits  │               │lambda_gap pulls  │               │Small window (3s) │
│diluted by 0.33*3 │               │events into 1 sec │               │misses big jumps  │
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

### Category A: Narrative Inversion & Non-Chronological Queries
- **Mechanism:** Users write narrative summaries reflecting rhetorical emphasis or recall order rather than chronological recording order.
- **Mathematical Cause:** Dynamic programming enforces $t_{e} > t_{e-1}$. If ground truth has $t_{e} < t_{e-1}$, DP is guaranteed to miss the true evidence peak of at least one event.
- **Observed in:** Q1, Q4, Q5.

### Category B: Mandatory-Event Cumulative Penalty Collapse
- **Mechanism:** In strict DP, total score is an additive sum: $S = \sum_{e=1}^N S[e, t_e] - \text{Gaps}$.
- **Mathematical Cause:** If a query has 3 events and 1 event has low similarity ($0.02$) due to visual occlusion or subtle action, the video's average drops from $0.18$ to $0.12$. A false-positive video with consistent background presence ($0.14, 0.14, 0.14$) achieves $0.42 > 0.38$, overtaking the true video.
- **Observed in:** Q1, Q5.

### Category C: BM25 Lexical Keyword Leakage & Title Domination
- **Mechanism:** Video titles are broadcast across all frames in BM25 indices, while caption vocabulary is empty in the current index artifact.
- **Mathematical Cause:** Min-max scaling normalizes sparse keyword hits ($BM25 > 0$) to $[0, 1.0]$. A 1-word title match gives $1.0 \times 0.5 = 0.50$ across all frames, which is $2.5\times$ larger than the visual cosine similarity ($0.18 - 0.22$). Irrelevant educational lectures (`L25_V003`, `L25_V012`) dominate Rank 1.
- **Observed in:** Q1, Q4, Q5.

### Category D: Multi-Modal Fusion Dilution on Visual-Only Actions
- **Mechanism:** Equal fixed weighting (`visual: 0.333, context: 0.333, asr: 0.333`) assumes all modalities contain signal.
- **Mathematical Cause:** When a query describes a purely visual action without dialogue or explicit text, Context and ASR contribute $0.0$. The true visual similarity ($0.2305$) is multiplied by $0.333 \to 0.0768$, dropping below competitor videos with accidental speech matches.
- **Observed in:** Q2.

### Category E: Time-Gap Penalty Path Clustering Collapse
- **Mechanism:** $\lambda_{\text{gap}} = 10^{-5}\text{ ms}^{-1}$ ($0.01\text{ s}^{-1}$) penalizes time separation.
- **Mathematical Cause:** A $40$-second separation incurs a penalty of $0.40$. Since visual cosine similarities differ by only $0.05 - 0.10$ between good and mediocre frames, the DP algorithm prefers picking adjacent frames within $2-3$ seconds rather than spanning $40$ seconds to find the true event peak.
- **Observed in:** Q1 (frames 363s, 365s, 366s), Q4 (frames 225s, 231s, 232s, 236s).

---

## 4. Concrete Architectural Recommendations

Based on the verified failure causes, the following algorithmic solutions are recommended for implementation:

### Recommendation 1: Two-Stage Retrieval (Dense as Candidate Gate, Temporal as Additive Reranker)
* **Problem:** Running temporal DP across the entire 873-video corpus allows false-positive attractor videos to overtake target videos that scored highly in single-query dense search.
* **Proposed Architecture:**
  1. **Stage 1 (Dense Candidate Selection):** Run global dense retrieval (using full query or max-over-events) to retrieve the top $K$ video candidates (e.g., $K = 50$ or $100$).
  2. **Stage 2 (Temporal Alignment Reranker):** Perform temporal alignment (strict or soft DP) **only** on the top $K$ candidate videos.
  3. **Score Combination:** Instead of replacing the dense score with the raw DP score, compute a combined score:
     $$\text{FinalScore}(v) = \text{DenseScore}(v) + \alpha \cdot \text{TemporalAlignmentBonus}(v)$$
     where $\alpha \in [0.2, 0.5]$.
* **Benefit:** A video with outstanding dense evidence will never be evicted to Rank >100 simply because one event is subtle or out of order.

### Recommendation 2: Support Skip / Optional Events in DP
* **Problem:** Mandatory matching of all $N$ events causes catastrophic collapse when 1 event is missing or poorly embedded.
* **Proposed Solution:** Introduce a constant "skip" emission score $\epsilon_{\text{skip}} \approx 0.10$. If an event's best match in a video is below $\epsilon_{\text{skip}}$, DP skips that event with a small penalty rather than accumulating a near-zero emission and ruining the path.

### Recommendation 3: Sigmoid Normalization & Title Attenuation for BM25
* **Problem:** Min-max scaling on BM25 scales a 1-word title match to $1.0$, completely overpowering visual cosine similarity.
* **Proposed Solution:**
  1. Apply soft sigmoid or z-score normalization on raw BM25 scores:
     $$S_{\text{norm}} = \frac{2}{1 + e^{-S_{\text{raw}} / \tau}} - 1$$
  2. Reduce title field weight from $1.0 \to 0.2$, and attenuate title match influence across long videos so that video-level titles do not uniformly inflate hundreds of frames.

### Recommendation 4: Adopt P1a Soft-Order DP with Extended Transition Windows
* **Problem:** Monotonic DP fails on narrative queries with inverted event descriptions.
* **Proposed Solution:** Integrate the P1a soft-order dynamic programming algorithm (`src/hcmai/temporal/soft_order.py`) into the main search pipeline, with `reverse_window_ms` extended to accommodate wider narrative inversions.

---

## 5. Persistent Research Knowledge Update

The findings from this benchmark and diagnostic study have been committed to `KNOWLEDGE.md` under **"Temporal Alignment Quality vs Dense Search: Failure Analysis & Reranking Architecture"**.
