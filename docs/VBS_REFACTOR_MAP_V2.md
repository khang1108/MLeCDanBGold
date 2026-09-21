# VBS refactor map v2

| Area | Canonical path | Status |
|---|---|---|
| MP4 discovery / frames | `offline/corpus/` | canonical |
| Dataset + API roots | `offline/config.py` | canonical |
| Model API URL selection | `offline/clients/` | canonical |
| Stage runner / state | `offline/pipeline/` | canonical |
| Caption/OCR/Object/ASR artifacts | `offline/enrichment/` | retained |
| Dense/index artifacts | `offline/indexes/` + indexing scripts | retained |
| HTTP transport contracts | `llm/contracts/` | canonical |
| Model API routers | `llm/server/routers/` | split by capability |
| Core embedding app | `HCMAI_LLM_PROFILE=core` | canonical |
| Single-A6000 worker | `HCMAI_LLM_PROFILE=gpu` + `HCMAI_GPU_TASK` | canonical |
| `offline/vbs/` | compatibility re-export | transitional |
| `offline/ingestion/` | archive/S3 legacy | not used by V3C runner |

## Old -> new concepts

```text
data/videos                  -> /home/phuckhang/hdd/V3C/videos
archive/source/staging       -> removed from V3C execution path
published/                   -> direct atomic local artifacts
HCMAI_INFERENCE_BASE_URL     -> VBS_CORE_API_URL + VBS_GPU_API_URL
/embeddings                  -> /v1/embeddings/text
Qwen3-VL-8B                  -> Qwen3-VL-2B paper profile
S3/public ASR audio URL      -> multipart /v1/transcripts/asr-file
one enrichment router file   -> captions.py / ocr.py / objects.py
```
