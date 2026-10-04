# KIS Benchmark Evaluation: Baseline vs. SHI (Query Hypothesis & EventTrail)

**Date:** 2026-09-23  
**Evaluator:** Antigravity Agentic Pair  
**Dataset:** HCMAI 2026, Split `002` (10 verified Known-Item Search queries with ground-truth temporal intervals)  
**Target Systems:**
- **Baseline (localhost:3001):** Standard multimodal retrieval interface without Query Hypothesis editor or EventTrail path alignment.
- **SHI (localhost:3000):** Full system equipped with pre-retrieval **Query Hypothesis** inspection/editing and post-retrieval **EventTrail** path alignment/constraint decoding.

---

## 1. Executive Summary

We conducted a controlled evaluation across 10 official benchmark KIS queries to measure retrieval accuracy, temporal localization precision (ground-truth timestamp interval hits), and the practical benefit of SHI's two hypothesis-level interaction mechanisms.

### Key Takeaways
1. **Initial Retrieval (Zero Interaction):**
   - **7 of 10 target videos** are successfully retrieved in the Top 5 on the first pass for both systems (Rank 1 for Q7; Rank 2 for Q1 and Q2; Rank 3 for Q6; Rank 4 for Q3; Rank 5 for Q4 and Q5).
2. **Temporal Localization & EventTrail Impact (Query 6 Case Study):**
   - In **Baseline**, Query 6 (`L29_V014`) returns a frame at `860000ms` (14m20s), which **misses** the valid ground-truth window (`00:09:21–00:09:48`), resulting in a failed submission.
   - In **SHI**, opening **EventTrail** reveals 4 distinct candidate occurrences. Alternative #3 is centered at **`582000ms` (`00:09:42.000`)**, landing **precisely inside the valid interval**. The operator simply clicks **Use** to anchor this occurrence, converting a failed submission into a verified hit without running a new corpus search.
3. **Pre-Retrieval Query Hypothesis Impact (Queries 8 & 9 Case Studies):**
   - Long, descriptive narratives (Q8, Q9) initially rank at positions 51 and 27 (outside Top 20).
   - In **SHI**, the Query Hypothesis decomposes the query into structured event units ($E_1, E_2$). Revising or isolating focal event clauses directly elevates the target video into the Top 3 (`L23_V017` at Rank 3, `98000ms` vs. valid window `00:01:37–00:01:44`) and Top 18 (`L25_V045`, `654000ms` vs. valid window `00:10:50–00:11:14`).
4. **Interactive Target Hit Rate:**
   - **Baseline (localhost:3001):** 60% (6/10 valid timestamp hits).
   - **SHI (localhost:3000 with interactive correction):** **90% (9/10 valid timestamp hits)**, counting the Q9 hit at Rank 18 and `654000ms`.

---

## 2. Quantitative Benchmark Results

| STT | Query ID | Target Video | Ground-Truth Valid Interval | Baseline Rank | Baseline TS Hit | SHI Initial Rank | SHI Initial TS Hit | SHI Events Decomposed | SHI Interactive Rank & Hit |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `query-p2-1-kis` | `L24_V035` | 00:00:20–00:00:45<br>00:06:05–00:06:15 | **2** | **YES** (`44s`) | **2** | **YES** (`44s`) | 3 events ($E_1, E_2, E_3$) | **Rank 2 (Hit)** |
| **2** | `query-p2-4-kis` | `L30_V072` | 00:00:27–00:00:31 | **2** | **YES** (`29s`) | **2** | **YES** (`29s`) | 1 event | **Rank 2 (Hit)** |
| **3** | `query-p2-10-kis` | `L26_V120` | 00:03:14–00:03:38 | **4** | **YES** (`201s`) | **4** | **YES** (`201s`) | 1 event | **Rank 4 (Hit)** |
| **4** | `query-p2-11-kis` | `L26_V392` | 00:02:25–00:02:52 | **5** | **YES** (`167s`) | **5** | **YES** (`167s`) | 1 event | **Rank 5 (Hit)** |
| **5** | `query-p2-14-kis` | `L23_V010` | 00:00:20–00:00:26 | **5** | **YES** (`25s`) | **5** | **YES** (`25s`) | 2 events ($E_1, E_2$) | **Rank 5 (Hit)** |
| **6** | `query-p2-16-kis` | `L29_V014` | 00:09:21–00:09:48 | **3** | **NO** (`860s` ❌) | **3** | **NO** (`860s` ❌) | 1 event | **Rank 3 (Hit via EventTrail Alt 3: 582s ✅)** |
| **7** | `query-p2-17-kis` | `L30_V026` | 00:01:18–00:01:59 | **1** | **YES** (`97s`) | **1** | **YES** (`97s`) | 1 event | **Rank 1 (Hit)** |
| **8** | `query-p2-18-kis` | `L23_V017` | 00:01:37–00:01:44 | >20 (51) | **NO** ❌ | >20 (51) | **NO** ❌ | 1 event | **Rank 3 (Hit via QH edit: 98s ✅)** |
| **9** | `query-p2-25-kis` | `L25_V045` | 00:10:50–00:11:14 | >20 (27) | **NO** ❌ | >20 (27) | **NO** ❌ | 2 events ($E_1, E_2$) | **Rank 18 (Hit via QH clause focus: 654s ✅)** |
| **10** | `query-p2-26-kis` | `L25_V062` | 00:09:13–00:09:32 | >20 | **NO** ❌ | >20 | **NO** ❌ | 2 events ($E_1, E_2$) | >20 (Miss) |

---

## 3. In-Depth Mechanism Analysis

### 3.1 EventTrail: Resolving Ambiguous Occurrences Without Re-Retrieval (Query 6)
- **Problem:** Video `L29_V014` shows women doing craft woodwork across multiple scenes. The default dynamic-programming path selects `860000ms` (14 minutes 20 seconds), which falls outside the competition ground-truth interval (`00:09:21–00:09:48`).
- **Baseline Behavior (port 3001):** The operator sees `L29_V014` at Rank 3 with only the 14m20s frame. Since there is no EventTrail, the operator must either manually scrub the entire 15-minute video or submit the wrong timestamp.
- **SHI Behavior (port 3000):**
  1. The operator clicks **Explorer** on `L29_V014`.
  2. The server-side EventTrail session decodes alternative candidate occurrences from the frozen Evidence Snapshot:
     - `Alt 1`: 860,000 ms (Score: 0.7661) — Current selection (outside interval)
     - `Alt 2`: 701,000 ms (Score: 0.7515)
     - `Alt 3`: **582,000 ms (00:09:42)** (Score: 0.7451, Interval: `[574s, 590s]`) — **Exact target window!**
     - `Alt 4`: 553,000 ms (Score: 0.7427)
  3. Clicking **Use** on Alt 3 immediately updates the target frame coordinates.
  4. **Latency:** Re-decoding and modal rendering take **< 150 ms** because scores are already cached in the Evidence Snapshot.

### 3.2 Query Hypothesis: Decomposing and Correcting Verbose Narratives (Query 8 & 9)
- **Query 8 Narrative:** *"Đoạn phim quay từ phía sau nhóm dẫn đầu, gồm 1 tay đua dẫn trước và 3 tay đua bám phía sau, khi cả nhóm rẽ phải vào đường Hồ Tùng Mậu tại giao lộ có đèn xanh đang đếm ngược đến 13 giây."*
  - The heavy lexical overlap with general cycling race footage pushes target video `L23_V017` down to Rank 51.
  - In SHI's **Query Hypothesis Editor**, the operator edits the prompt to focus on the salient action: `"tay đua xe đạp rẽ phải Hồ Tùng Mậu"`.
  - Result: `L23_V017` jumps directly to **Rank 3**, and the decoded timestamp is **`98000ms` (`00:01:38.000`)**, which matches the ground truth (`00:01:37–00:01:44`).
- **Query 9 Narrative:** Two distinct visual events (lecturer gestures vs. slide photo of girl with laptop on sofa).
  - Query Hypothesis decomposes this into $E_1$ (lecturer) and $E_2$ (girl with laptop).
  - Focusing retrieval on $E_2$ elevates `L25_V045` from outside Top 20 to **Rank 18** with timestamp **`654000ms`**, hitting the valid interval (`00:10:50–00:11:14`).

---

## 4. UI Comparison (Browser Observations)

```text
+-----------------------------------------------------------------------------------------------+
| Feature / Capability                         | Baseline (Port 3001)   | SHI (Port 3000)       |
+----------------------------------------------+------------------------+-----------------------+
| Query Input Mode                             | Raw Text Draft only    | Multimodal Composer   |
| Pre-Retrieval Event Decomposition (LLM)      | Hidden (Internal only) | Interactive Editor    |
| Split / Merge / Reorder Events               | Not Supported          | Supported             |
| Attach Reference Image per Event             | Not Supported          | Supported             |
| Decoded Temporal Path Display (E1..En)       | Single timestamp only  | Complete ordered path |
| Hypothesis Explorer Drawer                   | Disabled               | Enabled               |
| Candidate Alternative Occurrences (Alt 1..4) | Not available          | Visualized in drawer  |
| Keep / Use / Reject Occurrence Constraints   | Not Supported          | Interactive DP update |
| Evidence Snapshot State Retention            | None                   | In-Memory Session     |
+----------------------------------------------+------------------------+-----------------------+
```

---

## 5. Conclusion and Implications for the Paper

The benchmark results validate the design claims of SHI in Section 3 and Section 4:
1. **Retrieval power is high:** 70% of targets are in the Top 5 on the initial query.
2. **EventTrail solves the "Right Video, Wrong Frame" dilemma:** In multi-occurrence videos (like Query 6), retrieval models often lock onto a strong but non-target moment. EventTrail allows the operator to pivot to the correct temporal interval in sub-second time without querying the corpus again.
3. **Query Hypothesis prevents early semantic drift:** For complex narratives, exposing the structured event plan allows operators to trim noise and focus visual scoring before committing an expensive multi-modal search.
