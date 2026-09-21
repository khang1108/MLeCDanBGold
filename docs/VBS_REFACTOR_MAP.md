# VBS local refactor map

## Target execution path

```text
data/videos/*.mp4
  -> scripts.corpus.prepare_vbs_frames
  -> data/frames/<video_id>/*.jpg
  -> data/artifacts/frames.parquet
  -> Caption API / OCR API / Objects API / ASR API
  -> data/artifacts/{captions,ocr,asr,...}
  -> FrameContext
  -> remote embedding API + local FAISS build
  -> data/indexes/{visual,context,asr}
  -> hcmai runtime reads data/ only
```

## File ownership decisions

| Existing area | VBS action | Reason |
| --- | --- | --- |
| `offline/ingestion/custom_pipeline/archive.py` | remove from VBS path | no ZIP/archive downloads |
| `offline/ingestion/custom_pipeline/commit.py` | remove from VBS path | no staged/published batch lifecycle |
| `offline/ingestion/custom_pipeline/finalize.py` | remove from VBS path | no archive batch finalization |
| `offline/ingestion/custom_pipeline/shards.py` | remove from VBS path | old archive-scoped sharding |
| `offline/ingestion/custom_pipeline/state.py` | replace for VBS | use small local `data/state/` checkpoints |
| `offline/ingestion/s3.py` | never import from VBS path | local data plane only |
| `offline/keyframes/keyframes_extraction/**` | keep legacy, do not invoke | tightly coupled to `source/staging/published` |
| `scripts/corpus/extract_custom_keyframes.py` | superseded | media-info/yt-dlp/native lifecycle no longer needed |
| `scripts/corpus/prepare_custom_pipeline.py` | superseded | archive-oriented orchestration |
| `scripts/corpus/materialize_custom_frames.py` | superseded | new frame producer writes canonical artifact directly |
| `offline/enrichment/caption/**` | keep | already supports direct image upload to model API |
| `offline/enrichment/ocr/**` | keep | already supports direct image upload to model API |
| `offline/enrichment/transcripts/**` | keep + new upload adapter | VBS uploads locally extracted FLAC directly; old HTTPS reference endpoint remains compatibility-only |
| `offline/enrichment/context/**` | keep | deterministic local derivation |
| `offline/enrichment/object_detection.py` | keep artifact logic | canonical parsing/materialization is reused; VBS inference uses the new remote objects endpoint |
| `offline/embeddings/**` | keep | useful builder/checkpoint logic |
| `offline/indexes/**` | keep | local FAISS publication remains correct |
| `scripts/indexing/build_retrieval_indexes.py` | keep, never use `--s3` | remote embeddings + local index output already supported |
| `src/hcmai/corpus/**` | keep | runtime contracts remain valid |
| `src/hcmai/retrieval/**` | keep | index formats remain valid |
| `src/hcmai/common/config.py` | change defaults | runtime should read `data/artifacts` + `data/indexes` |

## Old path -> new path

| Old | New |
| --- | --- |
| `runs/custom-raw1fps-v1/source/...` | `data/videos/...` |
| `runs/.../staging/<video>/images/...` | `data/frames/<video>/...` |
| `runs/.../published/...` | removed |
| `artifacts/frame_store/frames.parquet` | `data/artifacts/frames.parquet` |
| `artifacts/enrichment/caption/...` | `data/artifacts/captions/...` |
| `artifacts/enrichment/ocr/...` | `data/artifacts/ocr/...` |
| `artifacts/enrichment/objects/...` | `data/artifacts/objects/...` |
| `artifacts/enrichment/transcripts/...` | `data/artifacts/asr/...` |
| `artifacts/enrichment/context*/...` | `data/artifacts/context/...` |
| `artifacts/indexes/visual` | `data/indexes/visual` |
| `artifacts/indexes/context*` | `data/indexes/context` |
| `artifacts/indexes/asr_segments` | `data/indexes/asr` |

## Naming rules

- `video_id = Path(mp4).stem`
- frame JPEG: `frames/<video_id>/<sample_index:06d>.jpg`
- internal `frame_id = <video_id>_<sample_index:06d>`
- VBS/DRES submission uses the retained `timestamp_ms` and exact `video_id`
- no path contains dataset-specific AIC/BTC/archive vocabulary in the VBS path

## New VBS-specific entry points

- `scripts/corpus/prepare_vbs_frames.py`: local MP4 -> sampled frames + canonical FrameStore.
- `scripts/enrichment/prepare_vbs_objects.py`: local JPEG batch -> remote YOLOE -> local object artifacts.
- `scripts/enrichment/prepare_vbs_transcripts.py`: local MP4 -> temporary FLAC -> remote ASR -> local transcript artifacts.
- `/v1/enrichment/objects`: multipart object-detection API.
- `/v1/transcripts/asr-file`: multipart ASR API; eliminates S3/public audio URLs.
