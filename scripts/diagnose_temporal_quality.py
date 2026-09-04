"""Diagnostic script for Temporal Alignment Quality vs Dense Search.

Executes API calls against http://127.0.0.1:8000 (Search & TRAKE)
and http://127.0.0.1:8100 (Embeddings), slices mmap vectors (<2MB RAM),
and computes R0, R1, R2 diagnostics across 5 benchmark queries.
"""

import json
import math
import os
import sys
import time
import urllib.request
import numpy as np

sys.path.insert(0, ".worktrees/p1a-soft-order/src")
from hcmai.temporal.soft_order import align_video_soft_order, SoftOrderParams
from hcmai.temporal.dp import align_video
from hcmai.retrieval.retriever.video_scores import VideoEventScores

BENCHMARK_QUERIES = [
    {
        "id": "Q1_pumpkin_lion_dance",
        "target_video_id": "L24_V035",
        "full_query_vi": "Nhóm 5 người đang chơi đùa bên cạnh một con vật màu vàng, một trong số đó đã mang một vật trông như trái bí đỏ đi giấu, người đàn ông thức dậy không thấy quả bí đỏ đâu nên đánh thức con vật dậy.",
        "events_vi": [
            "Nhóm 5 người đang chơi đùa bên cạnh một con vật màu vàng",
            "Một trong số đó đã mang một vật trông như trái bí đỏ đi giấu",
            "Người đàn ông thức dậy không thấy quả bí đỏ đâu nên đánh thức con vật dậy"
        ],
        "events_en": [
            "group of five people playing beside a yellow animal",
            "one person takes an object resembling a pumpkin and hides it",
            "a man wakes up, notices the pumpkin is missing, and wakes the animal"
        ],
    },
    {
        "id": "Q2_lion_dance_ship",
        "target_video_id": "L24_V044",
        "full_query_vi": "Một chú lân (hay rồng/sư tử?) màu vàng nhảy hay rơi từ trên cao xuống, gần với mô hình chiếc tàu thủy nhỏ màu xanh dương.",
        "events_vi": [
            "Một chú lân màu vàng nhảy hay rơi từ trên cao xuống gần với mô hình chiếc tàu thủy nhỏ màu xanh dương"
        ],
        "events_en": [
            "a yellow lion dance costume jumps or falls from above near a small blue model ship"
        ],
    },
    {
        "id": "Q3_graffiti_rhino_monkeys",
        "target_video_id": "L21_V013",
        "full_query_vi": "Đoạn clip bắt đầu với cảnh một người đang dùng điện thoại chụp ảnh bức tranh hình tê giác trên tường, đoạn clip kết thúc với cảnh một người chụp ảnh các hình graffiti 3 chú khỉ trên một cây cầu.",
        "events_vi": [
            "Một người đang dùng điện thoại chụp ảnh bức tranh hình tê giác trên tường",
            "Một người chụp ảnh các hình graffiti 3 chú khỉ trên một cây cầu"
        ],
        "events_en": [
            "A person taking a photo with a smartphone of a rhino mural on a wall",
            "A person taking photos of graffiti of three monkeys on a bridge"
        ],
    },
    {
        "id": "Q4_western_fruits_trake",
        "target_video_id": "L27_V015",
        "full_query_vi": "Video về một khu vườn cây ăn trái ở miền Tây Nam Bộ có chuỗi liên tiếp các cảnh quay về 4 loại trái cây trong vườn: cảnh có trái sầu riêng, cảnh có trái măng cụt, cảnh có trái bưởi, cảnh có trái dâu bòn bon.",
        "events_vi": [
            "Cảnh có trái sầu riêng",
            "Cảnh có trái măng cụt",
            "Cảnh có trái bưởi",
            "Cảnh có trái dâu bòn bon"
        ],
        "events_en": [
            "A shot showing durian fruit on the tree",
            "A shot showing mangosteen fruit",
            "A shot showing pomelo or grapefruit",
            "A shot showing langsat or bonbon fruit clusters"
        ],
    },
    {
        "id": "Q5_cooking_galangal_flowers",
        "target_video_id": "L26_V254",
        "full_query_vi": "Một cô gái mặc tạp dề trắng đứng cạnh một lọ hoa riềng tía, cô gái mặc tạp dề trắng đặt bốn nguyên liệu X chưa xác định lên một đĩa trắng, cùng cô gái mặc tạp dề trắng cầm hai nguyên liệu X cùng loại lên, cô gái mặc tạp dề trắng nói chuyện với một người ngồi đối diện về món ăn sẽ nấu.",
        "events_vi": [
            "Một cô gái mặc tạp dề trắng đứng cạnh một lọ hoa riềng tía",
            "Cô gái mặc tạp dề trắng đặt bốn nguyên liệu X chưa xác định lên một đĩa trắng",
            "Cùng cô gái mặc tạp dề trắng cầm hai nguyên liệu X cùng loại lên",
            "Cô gái mặc tạp dề trắng nói chuyện với một người ngồi đối diện về món ăn sẽ nấu"
        ],
        "events_en": [
            "A woman wearing a white apron stands beside a vase of purple galangal flowers.",
            "A woman wearing a white apron places four unidentified ingredients X on a white plate.",
            "The same woman wearing a white apron holds two of the same unidentified ingredients X.",
            "The woman wearing a white apron talks with a person seated opposite her about the dish they will cook."
        ],
    },
]

def call_api(endpoint, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error calling {endpoint}: {e}")
        return None

def embed_texts_siglip(texts):
    data = json.dumps({"source": "visual", "texts": texts}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8100/v1/embeddings/text",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return np.array(res["embeddings"], dtype=np.float32)

def main():
    print("Loading posting index metadata...")
    with open("artifacts/indexes/visual/posting_video_ids.json") as f:
        video_ids = json.load(f)
    offsets = np.load("artifacts/indexes/visual/posting_offsets.npy")
    timestamps = np.load("artifacts/indexes/visual/timestamps.npy")
    vectors = np.load("artifacts/indexes/visual/vectors.npy", mmap_mode="r")

    results = []

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        tgt = q["target_video_id"]
        print(f"\n=======================================================")
        print(f"Diagnosing {qid} (Target: {tgt})")
        print(f"=======================================================")

        # 1. R0 Dense Only (Single query)
        resp_r0 = call_api("/api/v1/search", {
            "query": q["full_query_vi"],
            "top_k": 100,
            "use_dense": True,
            "use_bm25": False,
        })
        r0_vids = [r["video_id"] for r in resp_r0["results"]] if resp_r0 else []
        r0_rank = r0_vids.index(tgt) + 1 if tgt in r0_vids else ">100"
        r0_score = resp_r0["results"][r0_vids.index(tgt)]["score"] if tgt in r0_vids else None
        r0_top1 = r0_vids[0] if r0_vids else None
        r0_top1_score = resp_r0["results"][0]["score"] if resp_r0 and resp_r0["results"] else None

        # 2. R0 Hybrid (Dense + BM25)
        resp_r0_hyb = call_api("/api/v1/search", {
            "query": q["full_query_vi"],
            "top_k": 100,
            "use_dense": True,
            "use_bm25": True,
        })
        r0_hyb_vids = [r["video_id"] for r in resp_r0_hyb["results"]] if resp_r0_hyb else []
        r0_hyb_rank = r0_hyb_vids.index(tgt) + 1 if tgt in r0_hyb_vids else ">100"
        r0_hyb_score = resp_r0_hyb["results"][r0_hyb_vids.index(tgt)]["score"] if tgt in r0_hyb_vids else None
        r0_hyb_top1 = r0_hyb_vids[0] if r0_hyb_vids else None

        # 3. R1 Strict Temporal DP (Dense Only)
        resp_r1 = call_api("/api/v1/trake", {
            "events": q["events_vi"],
            "top_k": 100,
            "use_dense": True,
            "use_bm25": False,
        })
        r1_vids = [p["video_id"] for p in resp_r1["paths"]] if resp_r1 else []
        r1_rank = r1_vids.index(tgt) + 1 if tgt in r1_vids else ">100"
        r1_path = resp_r1["paths"][r1_vids.index(tgt)] if tgt in r1_vids else None
        r1_score = r1_path["score"] if r1_path else None
        r1_top1 = r1_vids[0] if r1_vids else None
        r1_top1_score = resp_r1["paths"][0]["score"] if resp_r1 and resp_r1["paths"] else None

        # 4. R1 Hybrid Strict Temporal DP (Dense + BM25)
        resp_r1_hyb = call_api("/api/v1/trake", {
            "events": q["events_vi"],
            "top_k": 100,
            "use_dense": True,
            "use_bm25": True,
        })
        r1_hyb_vids = [p["video_id"] for p in resp_r1_hyb["paths"]] if resp_r1_hyb else []
        r1_hyb_rank = r1_hyb_vids.index(tgt) + 1 if tgt in r1_hyb_vids else ">100"
        r1_hyb_score = resp_r1_hyb["paths"][r1_hyb_vids.index(tgt)]["score"] if tgt in r1_hyb_vids else None
        r1_hyb_top1 = r1_hyb_vids[0] if r1_hyb_vids else None

        # 5. Extract Event Visual Vectors & Compute Detailed Scores for Target and Top1
        emb_events = embed_texts_siglip(q["events_vi"])
        n_events = len(q["events_vi"])

        def get_video_scores(vid):
            if vid not in video_ids:
                return None
            idx = video_ids.index(vid)
            s, e = offsets[idx], offsets[idx+1]
            v_vecs = vectors[s:e]
            v_ts = timestamps[s:e]
            sc = np.dot(emb_events, v_vecs.T)
            return sc, v_ts, e - s

        tgt_sc, tgt_ts, tgt_nframes = get_video_scores(tgt)
        tgt_ev_max = [float(np.max(tgt_sc[i])) for i in range(n_events)]
        tgt_ev_argmax_ts = [int(tgt_ts[np.argmax(tgt_sc[i])]) for i in range(n_events)]

        top1_sc, top1_ts, top1_nframes = get_video_scores(r1_top1) if r1_top1 else (None, None, 0)
        top1_ev_max = [float(np.max(top1_sc[i])) for i in range(n_events)] if top1_sc is not None else []
        top1_ev_argmax_ts = [int(top1_ts[np.argmax(top1_sc[i])]) for i in range(n_events)] if top1_ts is not None else []

        # 6. R2 Soft-order P1a DP evaluation
        # Run soft order on target video and top candidate videos
        candidate_vids = list(dict.fromkeys(r1_vids[:50] + [tgt]))
        soft_scores = {}
        strict_scores = {}
        for cvid in candidate_vids:
            csc, cts, cnf = get_video_scores(cvid)
            if csc is None or cnf < n_events:
                continue
            v_obj = VideoEventScores(
                video_id=cvid,
                frame_idx=tuple(range(cnf)),
                frame_ids=tuple(f"{cvid}_{i}" for i in range(cnf)),
                timestamps_ms=tuple(int(t) for t in cts),
                scores=csc.astype(np.float32)
            )
            s_paths = align_video_soft_order(v_obj, params=SoftOrderParams(reverse_window_ms=10000))
            if s_paths:
                soft_scores[cvid] = s_paths[0].path.score
            st_paths = align_video(v_obj)
            if st_paths:
                strict_scores[cvid] = st_paths[0].score

        sorted_soft = sorted(soft_scores.items(), key=lambda x: x[1], reverse=True)
        soft_vids = [x[0] for x in sorted_soft]
        r2_rank = soft_vids.index(tgt) + 1 if tgt in soft_vids else ">50"
        r2_score = soft_scores.get(tgt)

        report_item = {
            "id": qid,
            "target_video_id": tgt,
            "n_events": n_events,
            "r0_dense_rank": r0_rank,
            "r0_dense_score": r0_score,
            "r0_top1": r0_top1,
            "r0_top1_score": r0_top1_score,
            "r0_hyb_rank": r0_hyb_rank,
            "r0_hyb_score": r0_hyb_score,
            "r0_hyb_top1": r0_hyb_top1,
            "r1_dense_rank": r1_rank,
            "r1_dense_score": r1_score,
            "r1_top1": r1_top1,
            "r1_top1_score": r1_top1_score,
            "r1_hyb_rank": r1_hyb_rank,
            "r1_hyb_score": r1_hyb_score,
            "r1_hyb_top1": r1_hyb_top1,
            "r2_soft_rank": r2_rank,
            "r2_soft_score": r2_score,
            "target_event_max_scores": tgt_ev_max,
            "target_event_argmax_timestamps": tgt_ev_argmax_ts,
            "top1_event_max_scores": top1_ev_max,
            "top1_event_argmax_timestamps": top1_ev_argmax_ts,
            "target_r1_path": r1_path,
        }
        results.append(report_item)

        print(f"R0 (Dense 1-Query): Rank = {r0_rank}, Score = {r0_score}")
        print(f"R0_hyb (Dense+BM25): Rank = {r0_hyb_rank}, Score = {r0_hyb_score}")
        print(f"R1 (Strict DP):     Rank = {r1_rank}, Score = {r1_score}")
        print(f"R1_hyb (Strict+BM25): Rank = {r1_hyb_rank}, Score = {r1_hyb_score}")
        print(f"R2 (Soft DP P1a):   Rank = {r2_rank}, Score = {r2_score}")
        print(f"Target Event Max: {tgt_ev_max}")
        print(f"Target Event Timestamps: {tgt_ev_argmax_ts}")
        print(f"Top 1 ({r1_top1}) Event Max: {top1_ev_max}")

        os.makedirs("docs/research", exist_ok=True)
        with open("docs/research/2026-09-04-benchmark-diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Save full json output
    os.makedirs("docs/research", exist_ok=True)
    with open("docs/research/2026-09-04-benchmark-diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nDiagnostics complete! Saved to docs/research/2026-09-04-benchmark-diagnostics.json")

if __name__ == "__main__":
    main()
