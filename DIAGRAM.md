# HCMAI System Architecture & Interactive Hypothesis Diagrams

This document specifies the complete system architecture and interactive workflow for the **HCMAI / Ho Chi Minh City AI Challenge (Multimodal Video Retrieval)** system (SHI / EventTrail).

Canonical frame identity is strictly preserved across every offline and online stage as:
$$\text{Canonical Identity} = (\mathtt{video\_id}, \mathtt{frame\_id}, \mathtt{frame\_idx}, \mathtt{timestamp\_ms})$$

---

## 1. End-to-End System Architecture (3-Tier Overview)

The architecture is organized into three distinct, cooperating layers:
1. **Offline Layer**: Multimodal enrichment, deterministic context fusion, and vector indexing.
2. **Search Layer**: Multi-event multimodal retrieval, candidate generation, RRF fusion, and DP temporal sequence alignment.
3. **Interactive Hypothesis Layer**: Dual interaction loops featuring pre-retrieval **Query Hypothesis** reformulation and post-retrieval **EventTrail** fast DP temporal re-decoding.

```mermaid
flowchart TB
%% ==========================================
%% STYLING DEFINITIONS
%% ==========================================
    classDef offlineData fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px,color:#01579b;
    classDef modelNode fill:#ede7f6,stroke:#7e57c2,stroke-width:1.5px,color:#4527a0;
    classDef vectorDb fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#004d40;
    classDef searchNode fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px,color:#e65100;
    classDef fusionNode fill:#fce4ec,stroke:#d81b60,stroke-width:1.5px,color:#880e4f;
    classDef interactiveNode fill:#f3e5f5,stroke:#8e24aa,stroke-width:1.5px,color:#4a148c;
    classDef actionNode fill:#e8f5e9,stroke:#43a047,stroke-width:1.5px,color:#1b5e20;
    classDef outputNode fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#f57f17;

%% ==========================================
%% LAYER 1: OFFLINE PREPARATION & INDEXING
%% ==========================================
    subgraph LAYER_OFFLINE["LAYER 1: OFFLINE ENRICHMENT & VECTOR INDEXING"]
        direction TB

        subgraph ingestion["Raw Corpus Ingestion"]
            direction LR
            raw_videos["Raw Videos (.mp4)"]
            keyframes["Keyframes Extracted<br/>(Canonical Frames)"]
            audio_track["Audio Stream (16kHz)"]
        end

        raw_videos --> keyframes
        raw_videos --> audio_track

        subgraph enrichment_models["Enrichment Specialist Models"]
            direction LR
            qwen_vl["Qwen3-VL-8B-Instruct<br/>(Dense Visual Captioning)"]
            florence["Florence-2-base-ft<br/>(OCR Region & Text Extraction)"]
            yoloe["YOLOE-26L (yoloe-26l-seg)<br/>(Object Detection & Counts)"]
            qwen_asr["Qwen3-ASR-1.7B-hf<br/>(+ PyAnnote Diarization)"]
        end

        keyframes --> qwen_vl
        keyframes --> florence
        keyframes --> yoloe
        audio_track --> qwen_asr

        subgraph intermediate_evidence["Structured Specialist Evidence"]
            direction LR
            captions["Frame Captions<br/>(80-token budget)"]
            ocr_text["Normalized OCR<br/>(80-token budget)"]
            objects["Object Summary<br/>(40-token budget)"]
            asr_segments["ASR Segments<br/>(Timestamped Timeline)"]
        end

        qwen_vl --> captions
        florence --> ocr_text
        yoloe --> objects
        qwen_asr --> asr_segments

        captions & ocr_text & objects --> frame_context["Deterministic FrameContext<br/>(frame-context-v1)"]

        subgraph offline_embedding["Representation & Embeddings"]
            direction LR
            bge_ctx["BGE-M3 Context Encoder<br/>(BAAI/bge-m3, 8192 tok)"]
            bge_asr["BGE-M3 ASR Encoder<br/>(BAAI/bge-m3, 8192 tok)"]
            siglip_vis["SigLIP2 Visual Encoder<br/>(google/siglip2-base-patch16-224)"]
        end

        frame_context --> bge_ctx
        asr_segments --> bge_asr
        keyframes --> siglip_vis

        subgraph vector_stores["Vector Databases (FAISS Flat-IP)"]
            direction LR
            db_context[("FrameContext VectorDB<br/>(Dense Context Index)")]
            db_asr[("ASR VectorDB<br/>(Dense Segment Index)")]
            db_visual[("Visual VectorDB<br/>(Keyframe Index)")]
        end

        bge_ctx --> db_context
        bge_asr --> db_asr
        siglip_vis --> db_visual
    end

%% ==========================================
%% LAYER 2: MULTIMODAL SEARCH & TEMPORAL ALIGNMENT
%% ==========================================
    subgraph LAYER_SEARCH["LAYER 2: MULTIMODAL SEARCH & TEMPORAL ALIGNMENT"]
        direction TB

        subgraph query_input["Multimodal Query Input"]
            query_text["Text Query / Ordered Events<br/>(E₁, E₂, ..., Eₙ)"]
            query_image["Query Images / Visual Examples<br/>(Optional Reference Frames)"]
        end

        subgraph query_encoders["Online Query Encoders"]
            direction LR
            bge_query["BGE-M3 Text Encoder<br/>(Event-level Dense Embedding)"]
            siglip_query["SigLIP2 Image Encoder<br/>(Visual Query Embedding)"]
        end

        query_text --> bge_query
        query_image --> siglip_query

        subgraph candidate_retrieval["Candidate Retrieval per Event"]
            direction LR
            cand_context["FrameContext Candidates<br/>(Dense Semantic Match)"]
            cand_asr["ASR Candidates<br/>(Timeline Speech Match)"]
            cand_visual["Visual Candidates<br/>(Cross-modal / Image Match)"]
        end

        bge_query --> cand_context
        bge_query --> cand_asr
        siglip_query --> cand_visual

        subgraph rrf_fusion["Modality Fusion & Scoring"]
            rrf["Reciprocal Rank Fusion (RRF)<br/>(k = 60, Multi-source Weights)"]
            score_matrix["Event-Frame Score Matrix<br/>(Per Candidate Video)"]
        end

        cand_context --> rrf
        cand_asr --> rrf
        cand_visual --> rrf
        rrf --> score_matrix

        subgraph temporal_dp["Temporal Sequence Alignment"]
            dp_aligner["Dynamic Programming (DP) Aligner<br/>(Order Preservation + Gap Penalty λ)"]
            retrieval_results["Retrieval Results<br/>(Ranked Videos & Aligned Event Paths)"]
        end

        score_matrix --> dp_aligner
        dp_aligner --> retrieval_results
    end

%% Inter-layer connections: Vector DBs feed into Candidate Searchers
    db_context -.->|Index Search| cand_context
    db_asr -.->|Index Search| cand_asr
    db_visual -.->|Index Search| cand_visual

%% ==========================================
%% LAYER 3: INTERACTIVE HYPOTHESIS & CORRECTION
%% ==========================================
    subgraph LAYER_INTERACTIVE["LAYER 3: INTERACTIVE HYPOTHESIS & CORRECTION"]
        direction TB

        subgraph pre_search["1. Pre-Retrieval: Query Hypothesis"]
            direction TB
            user_narrative["Operator Narrative / Request"]
            qh_generator["Query Hypothesis Generator<br/>(Grounded Event Decomposition)"]
            qh_editor["Query Hypothesis Editor<br/>(Split, Merge, Reorder, Edit, Attach Images)"]
            qh_revision["Immutable Query Revision<br/>(query_revision: v₁, v₂, ...)"]

            user_narrative --> qh_generator
            qh_generator --> qh_editor
            qh_editor -->|Apply Changes| qh_revision
        end

        qh_revision -->|Drives Retrieval| query_text
        qh_editor -.->|Optional Reference| query_image

        subgraph snapshot_store["Evidence Snapshot Cache"]
            evidence_snapshot["Evidence Snapshot<br/>(Immutable Event Scores, Candidate Matrices, Decoder Config)"]
        end

        retrieval_results -->|Cache Results| evidence_snapshot
        score_matrix -.->|Retain Raw Scores| evidence_snapshot

        subgraph post_search["2. Post-Retrieval: EventTrail Temporal Correction"]
            direction TB
            trail_viewer["Decoded Temporal Hypothesis<br/>(Current Aligned Event Path E₁→E₂→...→Eₙ)"]
            alternative_preview["Alternative Occurrence Preview<br/>(Complete-Path Preview for Focused Event)"]
            
            subgraph trail_actions["Interactive Operator Actions"]
                direction LR
                act_keep["Keep Occurrence<br/>(Anchor Current)"]
                act_use["Use Alternative<br/>(Anchor New)"]
                act_reject["Reject Region<br/>(Exclude Mode)"]
                act_undo["Undo Action<br/>(Rollback Constraint)"]
            end

            fast_redecoder["Fast Local DP Re-Decoder<br/>(Re-decode Path from Stored Evidence Snapshot in ms)"]
            updated_path["Updated Aligned Path & Rank<br/>(trail_revision: r₁, r₂, ...)"]
        end

        evidence_snapshot --> trail_viewer
        trail_viewer --> alternative_preview
        alternative_preview --> trail_actions
        trail_actions --> fast_redecoder
        evidence_snapshot -.->|Zero-Retrieval Latency| fast_redecoder
        fast_redecoder --> updated_path
        updated_path --> trail_viewer

        subgraph verification_submission["3. Verification & DRES Submission"]
            direction TB
            inspector["Synchronized Inspector<br/>(High-Res Keyframes & Video Playback)"]
            dres_client["DRES Submission Client<br/>(video_id, frame_idx, timestamp_ms)"]
            competition_eval["DRES Server Outcome<br/>(Authoritative Evaluator Feedback)"]

            inspector --> dres_client
            dres_client --> competition_eval
        end

        updated_path --> inspector
        retrieval_results -.->|Direct Inspection| inspector
        trail_actions -.->|Semantic Error Reformulation| qh_editor
    end

%% Class Assignments
    class raw_videos,keyframes,audio_track,captions,ocr_text,objects,asr_segments,frame_context offlineData;
    class qwen_vl,florence,yoloe,qwen_asr,bge_ctx,bge_asr,siglip_vis,bge_query,siglip_query modelNode;
    class db_context,db_asr,db_visual vectorDb;
    class query_text,query_image,cand_context,cand_asr,cand_visual searchNode;
    class rrf,score_matrix,dp_aligner fusionNode;
    class user_narrative,qh_generator,qh_editor,qh_revision,evidence_snapshot,trail_viewer,alternative_preview interactiveNode;
    class act_keep,act_use,act_reject,act_undo,fast_redecoder,updated_path actionNode;
    class retrieval_results,inspector,dres_client,competition_eval outputNode;
```

---

## 2. Layer Deep-Dives

### 2.1 Layer 1: Offline Enrichment & Indexing Pipeline

This offline layer processes raw competition video collections into structured, canonical evidence and builds high-speed vector indexes.

```mermaid
flowchart LR
    subgraph Data["1. Video Ingestion"]
        V["Raw Videos (.mp4)"]
        KF["Keyframes (.jpg)<br/>BTC Extracted"]
        AUD["Audio Track (16kHz WAV)"]
        V --> KF
        V --> AUD
    end

    subgraph Specialist["2. Specialist Enrichment Models"]
        QWEN["Qwen3-VL-8B-Instruct<br/>(max_tokens: 96, bf16)"]
        FLOR["Florence-2-base-ft<br/>(OCR with Regions, bf16)"]
        YOLO["YOLOE-26L<br/>(Custom 14k Vocab, conf: 0.20)"]
        ASR["Qwen3-ASR-1.7B-hf<br/>(+ PyAnnote Diarization)"]

        KF --> QWEN
        KF --> FLOR
        KF --> YOLO
        AUD --> ASR
    end

    subgraph Structuring["3. Evidence Structuring"]
        CAP["Caption Evidence<br/>Token budget: 80"]
        OCR["Normalized OCR<br/>Token budget: 80"]
        OBJ["Object Multiplicity<br/>Token budget: 40"]
        SEG["ASR Timeline Segments<br/>(Speaker, start_ms, end_ms)"]

        QWEN --> CAP
        FLOR --> OCR
        YOLO --> OBJ
        ASR --> SEG

        CTX["Deterministic FrameContext<br/>(Caption + OCR + Objects)"]
        CAP & OCR & OBJ --> CTX
    end

    subgraph Embeddings["4. Multi-Vector Embeddings"]
        BGE_C["BAAI/bge-m3<br/>(Text / Context)"]
        BGE_A["BAAI/bge-m3<br/>(Speech Segments)"]
        SIG["SigLIP2-base-patch16-224<br/>(Visual Representation)"]

        CTX --> BGE_C
        SEG --> BGE_A
        KF --> SIG
    end

    subgraph VectorDBs["5. Vector Databases (FAISS Flat-IP)"]
        DB_C[("fa:fa-database FrameContext DB<br/>artifacts/indexes/context_vi")]
        DB_A[("fa:fa-database ASR DB<br/>artifacts/indexes/asr_segments")]
        DB_V[("fa:fa-database Visual DB<br/>artifacts/indexes/visual")]

        BGE_C --> DB_C
        BGE_A --> DB_A
        SIG --> DB_V
    end

    classDef stage fill:#f8f9fa,stroke:#455a64,stroke-width:1px;
    classDef model fill:#ede7f6,stroke:#7e57c2,stroke-width:1.5px;
    classDef db fill:#e0f2f1,stroke:#00897b,stroke-width:2px;
    class QWEN,FLOR,YOLO,ASR,BGE_C,BGE_A,SIG model;
    class DB_C,DB_A,DB_V db;
```

#### Key Engineering Invariants in Offline Pipeline
- **Canonical Coordinate Preservation**: Every generated record preserves `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms`.
- **Specialist Evidence Preservation**: Source captions, OCR bounding boxes, object label multiplicities, and raw transcripts remain accessible in standalone parquet stores (`artifacts/corpus/`); they are **never** destructively overwritten by unified context.
- **ASR Timeline Integrity**: Speech is indexed as timestamped interval evidence and mapped onto frames within a maximum window of 2,000 ms to 5,000 ms.

---

### 2.2 Layer 2: Online Search, RRF Fusion & Temporal DP Alignment

Online search executes multi-event retrieval across text and visual streams, fuses modality ranks via Reciprocal Rank Fusion (RRF), and decodes optimal temporal paths using Dynamic Programming.

```mermaid
flowchart TD
    subgraph QueryIngest["1. Query Decomposition & Encoders"]
        TQuery["Text Query / Ordered Events<br/>E₁: 'A man in red enters'<br/>E₂: 'He picks up a suitcase'<br/>E₃: 'He leaves through the gate'"]
        IQuery["Image Query / Visual Example<br/>(Optional reference keyframe)"]
        
        BGE_Q["BGE-M3 Query Encoder"]
        SIG_Q["SigLIP2 Query Encoder"]

        TQuery --> BGE_Q
        IQuery --> SIG_Q
    end

    subgraph IndexSearch["2. Modality Index Search (Top-K per Event)"]
        DB_C[("FrameContext DB")]
        DB_A[("ASR DB")]
        DB_V[("Visual DB")]

        Cand_C["Context Candidates<br/>(Top-K frames per Eᵢ)"]
        Cand_A["ASR Candidates<br/>(Top-K segments per Eᵢ)"]
        Cand_V["Visual Candidates<br/>(Top-K visual frames per Eᵢ)"]

        BGE_Q --> Cand_C
        BGE_Q --> Cand_A
        SIG_Q --> Cand_V
        DB_C -.-> Cand_C
        DB_A -.-> Cand_A
        DB_V -.-> Cand_V
    end

    subgraph Fusion["3. Reciprocal Rank Fusion (RRF)"]
        RRF_Op["RRF Score Accumulation<br/>RRF(f) = Σ wₘ / (k + rankₘ(f))<br/>(k = 60)"]
        Cand_C --> RRF_Op
        Cand_A --> RRF_Op
        Cand_V --> RRF_Op

        Matrix["Event-Frame Score Matrix S[video, event, frame]<br/>Normalized dense multimodal evidence"]
        RRF_Op --> Matrix
    end

    subgraph DP["4. Temporal Path Decoding (Dynamic Programming)"]
        DP_Align["DP Viterbi Alignment Aligner<br/>Maximize: Σ Score(Eᵢ, fᵢ) - λ_gap · Δt(fᵢ, fᵢ₋₁)<br/>Subject to: t(f₁) < t(f₂) < ... < t(fₙ)"]
        Matrix --> DP_Align
        
        Results["Top Ranked Paths<br/>Path = [(E₁, f*₁), (E₂, f*₂), ..., (Eₙ, f*ₙ)]<br/>Canonical frame_idx & video_id coordinates"]
        DP_Align --> Results
    end

    classDef db fill:#e0f2f1,stroke:#00897b,stroke-width:2px;
    classDef comp fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px;
    classDef highlight fill:#fce4ec,stroke:#d81b60,stroke-width:2px;
    class DB_C,DB_A,DB_V db;
    class TQuery,IQuery,Cand_C,Cand_A,Cand_V comp;
    class RRF_Op,Matrix,DP_Align,Results highlight;
```

#### Dynamic Programming (DP) Path Scoring
Given $N$ query events $(E_1, \dots, E_N)$ and candidate frame occurrences $f \in \mathcal{F}_v$ for video $v$:
$$\text{Score}(P) = \sum_{i=1}^{N} S(E_i, f_i) - \lambda_{\text{gap}} \sum_{i=2}^{N} \max(0, t(f_i) - t(f_{i-1}) - \Delta t_{\min})$$
Subject to strict chronological order:
$$t(f_1) < t(f_2) < \dots < t(f_N)$$
Where $\lambda_{\text{gap}} = 10^{-5}$ prevents degenerate distant alignments while allowing natural video event progression.

---

### 2.3 Layer 3: Interactive Hypothesis Loops (Query Hypothesis & EventTrail)

Interactive Hypothesis exposes two intermediate decision points as mutable interaction objects without triggering costly index recalculation:
1. **Pre-Retrieval (Query Hypothesis)**: Inspect and edit the semantic decomposition.
2. **Post-Retrieval (EventTrail)**: Inspect and fix temporal alignment errors in milliseconds via stored evidence snapshots.

```mermaid
flowchart TB
    subgraph PreRetrieval["PRE-RETRIEVAL INTERACTION: Query Hypothesis"]
        direction TB
        RawQuery["User Input Narrative"]
        Decomp["Query Hypothesis Decomposition<br/>Ordered Sequence H_q = (E₁, E₂, ..., Eₙ)"]
        
        subgraph QHEditor["Query Hypothesis Editor Actions"]
            direction LR
            EditE["Edit Text / Span"]
            SplitE["Split Event"]
            MergeE["Merge Events"]
            ReorderE["Reorder Events"]
            AddImg["Attach Reference Image"]
        end

        PreviewQH["Side-by-Side Preview<br/>(Current vs. Proposed Decomposition)"]
        CommitQH["Commit Query Revision<br/>(Creates monotonic query_revision vₖ)"]

        RawQuery --> Decomp
        Decomp --> QHEditor
        QHEditor --> PreviewQH
        PreviewQH -->|User Approves| CommitQH
    end

    CommitQH -->|Executes Search| SearchEngine["Layer 2: Search Pipeline (RRF + DP)"]

    subgraph SnapshotStage["Evidence Snapshot (Immutable Search State)"]
        Snap[("Evidence Snapshot Cache<br/>- Preserves Event-Frame Score Matrix S[v, e, f]<br/>- Decoded candidate paths<br/>- Canonical frame metadata<br/>- Keyed by snapshot_id & revision")]
    end

    SearchEngine --> Snap

    subgraph PostRetrieval["POST-RETRIEVAL INTERACTION: EventTrail"]
        direction TB
        SelectVideo["Operator Selects Video Candidate from Ranked List"]
        Snap --> SelectVideo

        DispPath["Display Decoded Temporal Hypothesis<br/>E₁: [frame_idx: 1240, 00:49.6]<br/>E₂: [frame_idx: 1850, 01:14.0]<br/>E₃: [frame_idx: 3100, 02:04.0]"]
        SelectVideo --> DispPath

        FocusEvent["Operator Focuses on Ambiguous Event (e.g., E₂)"]
        DispPath --> FocusEvent

        AltPreview["Preview Temporally Distinct Alternatives<br/>(Complete valid chronological path decoded in preview)"]
        FocusEvent --> AltPreview

        subgraph TrailActions["EventTrail Correction Actions"]
            direction LR
            ActKeep["KEEP<br/>Anchor current f*"]
            ActUse["USE<br/>Anchor previewed f'"]
            ActReject["REJECT<br/>Exclude temporal window"]
            ActUndo["UNDO<br/>Rollback constraint"]
        end

        AltPreview --> TrailActions

        FastDP["Fast Local DP Re-Decoder<br/>(Zero retrieval overhead, ~5 ms re-alignment on Snapshot)"]
        TrailActions --> FastDP
        Snap -.->|Reads Stored Matrix| FastDP
        FastDP -->|Updates Path| DispPath
    end

    subgraph Finalize["Verification & Submission"]
        direction TB
        Inspect["Inspect Synchronized Video & High-Res Frame"]
        DispPath --> Inspect
        SubmitDRES["Submit to DRES Server<br/>(video_id, frame_idx, timestamp_ms)"]
        Inspect --> SubmitDRES
    end

    classDef qh fill:#f3e5f5,stroke:#8e24aa,stroke-width:1.5px;
    classDef snap fill:#e0f2f1,stroke:#00897b,stroke-width:2px;
    classDef trail fill:#e8f5e9,stroke:#43a047,stroke-width:1.5px;
    classDef action fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px;
    classDef submit fill:#fffde7,stroke:#fbc02d,stroke-width:2px;

    class RawQuery,Decomp,PreviewQH,CommitQH qh;
    class Snap snap;
    class SelectVideo,DispPath,FocusEvent,AltPreview,FastDP trail;
    class EditE,SplitE,MergeE,ReorderE,AddImg,ActKeep,ActUse,ActReject,ActUndo action;
    class Inspect,SubmitDRES submit;
```

---

## 3. Interactive Execution Sequence (Paper-Ready Protocol)

The sequence diagram below summarizes the end-to-end interactive retrieval lifecycle in SHI, highlighting the dual hypothesis-correction loops (Pre-retrieval Query Hypothesis and Post-retrieval EventTrail over Evidence Snapshots):

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Operator
    participant QH as Query Hypothesis
    participant Search as Search Pipeline (RRF + DP)
    participant Snap as Evidence Snapshot
    participant Trail as EventTrail
    participant DRES as Evaluation (DRES)

    %% PHASE 1: QUERY HYPOTHESIS
    rect rgb(245, 238, 248)
    Note over User,QH: Phase 1: Pre-Retrieval Query Hypothesis
    User->>QH: Input narrative query
    QH-->>User: Inferred ordered event sequence H_q = (E₁, E₂, ..., Eₙ)
    opt Structural Correction
        User->>QH: Edit / Split / Merge / Reorder / Attach image
        QH-->>User: Preview modifications & commit updated revision
    end
    end

    %% PHASE 2: SEARCH & SNAPSHOT
    rect rgb(255, 248, 230)
    Note over User,Snap: Phase 2: Multimodal Search & Snapshotting
    User->>Search: Execute search with active H_q
    Search->>Search: Multi-event retrieval, RRF fusion & DP alignment
    Search->>Snap: Cache score matrices & candidate paths (Evidence Snapshot)
    Search-->>User: Initial ranked video candidates & aligned event paths
    end

    %% PHASE 3: EVENTTRAIL CORRECTION
    rect rgb(235, 247, 238)
    Note over User,Trail: Phase 3: Post-Retrieval EventTrail Correction
    User->>Trail: Select candidate video & focus on event (e.g., E₂)
    Trail->>Snap: Read immutable evidence snapshot
    Trail-->>User: Preview complete-path alternatives
    alt Apply Temporal Constraint
        User->>Trail: Action (KEEP / USE anchor / REJECT temporal region)
        Trail->>Trail: Fast local DP re-decoding (<10 ms, zero index re-query)
        Trail-->>User: Updated chronological path & refreshed candidate score
    end
    end

    %% PHASE 4: VERIFICATION & SUBMISSION
    rect rgb(255, 253, 231)
    Note over User,DRES: Phase 4: Verification & Competition Submission
    Note over User,QH: Verify synchronized keyframe & video playback
    User->>DRES: Submit canonical coordinate (video_id, frame_idx, timestamp_ms)
    DRES-->>User: Authoritative evaluation outcome
    end
```

---

## 4. Key Architectural Invariants

1. **Strict Canonical Identity Guarantee**:
   - `frame_id`: Internal unique canonical hash across all indexes and stores.
   - `frame_idx`: Competition-facing frame coordinate for official evaluation submissions.
   - `video_id`: Source video identifier.
   - `timestamp_ms`: Canonical millisecond offset in source video.
   - Under no circumstances does keyframe array index or model rank replace `frame_idx`.

2. **Immutable Revisions & Snapshot Isolation**:
   - Every mutation in Query Hypothesis produces an incremental `query_revision`. Older search results are explicitly marked stale.
   - EventTrail actions create incremental `trail_revision` instances isolated within the captured `snapshot_id`.
   - Re-decoding in EventTrail operates strictly on preserved evidence snapshots, ensuring sub-10ms response times without issuing new vector database queries.

3. **Additive Multimodal Evidence**:
   - `FrameContext` provides high-density synthesized text for general semantics.
   - Specialist models (`Qwen3-VL`, `Florence-2`, `YOLOE-26L`, `Qwen3-ASR`) preserve their isolated evidence stores for independent ablation and high-precision filtering.
   - `ASR` evidence represents temporal timeline intervals rather than frame-native pixels.

4. **Deterministic Dynamic Programming**:
   - Path search maximizes multimodal evidence while enforcing forward chronological monotonicity and applying soft distance penalties $\lambda_{\text{gap}}$.
   - Multi-path decoding extracts alternative event paths with guaranteed minimum temporal separation.
