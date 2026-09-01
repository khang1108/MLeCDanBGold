# HCMAI 2026 — Multimodal Video Retrieval

HCMAI là hệ thống tìm kiếm video đa phương thức cho HCMAI 2026. Người dùng
nhập truy vấn tiếng Việt hoặc tiếng Anh; hệ thống tìm các keyframe phù hợp,
giữ lại bằng chứng Caption/OCR/Object/ASR, định vị theo thời gian và trả về
`video_id` cùng `frame_idx` hợp lệ cho bài thi.

README này mô tả profile đang dùng: BTC cung cấp keyframes, mapping và
objects; HCMAI tạo enrichment, embedding và index. Không cần trích xuất lại
keyframe từ video trong profile này.

## 1. Tổng quan hệ thống

```text
Browser (React :3000)
        |
        v
Local FastAPI backend (:8000)
  ├─ canonical FrameStore + keyframes + artifacts + FAISS indexes
  ├─ retrieval / fusion / temporal localization
  └─ HTTP requests for model inference only
        |
        v
Remote GPU inference API (ThunderCompute + Cloudflare)
  ├─ SigLIP2 visual embedding
  ├─ BGE-M3 text embedding
  ├─ Qwen3-VL reranker for explicit offline experiments
  └─ caption/OCR/ASR/diarization services (optional)
```

Máy local giữ dữ liệu tìm kiếm và frontend/backend. GPU VM chỉ chạy model
inference, vì vậy có thể tắt VM sau khi benchmark mà không mất corpus/index
local.

### Các identity bắt buộc

- `frame_id`: identity nội bộ, dùng để join và truy xuất artifact.
- `video_id`: video nguồn và identity dùng khi nộp bài.
- `frame_idx`: tọa độ frame chính thức, là số nguyên lấy từ BTC mapping.
- `timestamp_ms`: thời điểm của keyframe.

Không được thay `frame_idx` bằng thứ tự keyframe, số thứ tự filename hoặc
array index. Reranker và model provider cũng không được tự tạo lại identity.

## 2. Hệ thống xử lý như thế nào?

### 2.1. Chuẩn bị dữ liệu offline

```text
BTC keyframes + map_keyframes + BTC objects
                 |
                 v
        Canonical FrameStore
        (frames.parquet + manifest)
                 |
       +---------+----------+----------------+
       v                    v                v
   Caption                 OCR          Object import
       \                    |                /
        +-------------------+---------------+
                            v
                    FrameContext V1

Videos (nếu cần ASR) --> timestamped transcript segments

FrameStore + FrameContext + transcripts
                            |
                            v
              Visual / Context / ASR indexes
```

Các bước chính:

1. Import metadata BTC và mapping thành `artifacts/frame_store/frames.parquet`.
2. Sinh Caption, OCR và import Object evidence. Các artifact chuyên biệt vẫn
   được giữ riêng; `FrameContext` chỉ là view kết hợp có thể tái tạo.
3. Nếu task cần lời thoại, chạy ASR trên video để tạo transcript có timestamp.
4. Build embedding và FAISS index offline. Online serving không tự build lại
   corpus hoặc index lớn.

Lệnh enrichment chi tiết nằm trong [`scripts/README.md`](scripts/README.md).

### 2.2. Luồng KIS online

```text
query
  -> query encoding
  -> visual/context/ASR retrieval
  -> RRF fusion, giữ provenance và canonical frame_id
  -> deterministic ordered event-to-frame alignment
  -> KIS path projection + canonical materialization
  -> SearchMaterializer
  -> SearchResponse (Top-K frame results)
```

KIS endpoint:

```text
POST /api/v1/search
```

### 2.3. Luồng TRAKE online

```text
ordered events
  -> retrieval cho từng event
  -> shortlist video cùng nguồn
  -> dense event x frame rescoring
  -> monotonic temporal alignment
  -> ranked ordered frame-path submissions
```

TRAKE endpoint:

```text
POST /api/v1/trake
```

### 2.4. Các endpoint thường dùng

| Endpoint | Mục đích |
| --- | --- |
| `GET /health` | Kiểm tra backend, FrameStore, index và inference |
| `POST /api/v1/search` | KIS search |
| `POST /api/v1/trake` | TRAKE ordered-event alignment |
| `GET /api/v1/keyframes/{frame_id}` | Lấy canonical keyframe theo internal `frame_id` |
| `GET /api/v1/frames/{frame_id}/neighbors` | Lấy frame lân cận theo thời gian |
| `POST /api/v1/submit` | Tạo identity submission cho một frame |

## 3. Cấu trúc thư mục và artifact

```text
HCMAI_2026/
├── src/hcmai/                 Python package và FastAPI backend
├── frontend/                  React frontend
├── configs/                   baseline + unified prepare/model/S3 config
├── scripts/                   CLI chuẩn bị dữ liệu, build index, benchmark
├── data/
│   ├── keyframes/             BTC keyframes: <video_id>/<order>.jpg
│   ├── map_keyframes/         BTC mapping CSV
│   ├── objects/               BTC object JSON (nếu đã sync)
│   └── videos/                video nguồn; chỉ cần khi chạy ASR
├── artifacts/
│   ├── frame_store/           frames.parquet + manifest.json
│   ├── enrichment/            captions, OCR, objects, context, transcripts
│   └── indexes/               visual, context, asr_segments
├── thundercompute/            shared inference service and config
├── tests/                     backend tests
└── README.md
```

Các path quan trọng khi chạy backend:

| Thành phần | Path mặc định |
| --- | --- |
| Canonical metadata | `artifacts/frame_store/frames.parquet` |
| Keyframe asset root | `data` |
| Visual index | `artifacts/indexes/visual` |
| Context index | `artifacts/indexes/context` |
| ASR segment index | `artifacts/indexes/asr_segments` |
| Caption artifact | `artifacts/corpus/caption.parquet` |
| OCR artifact | `artifacts/corpus/ocr_frames.parquet` |
| Object artifact | `artifacts/corpus/object_frames.parquet` |
| Context artifact | `artifacts/corpus/context.parquet` |
| Transcript artifact | `artifacts/enrichment/transcripts/` |

Lưu ý: `HCMAI_DATASET_ROOT` là root để resolve ảnh. Với canonical
`image_path` hiện tại trỏ tới `data/keyframes/...`, giá trị đúng là `data`,
không phải `artifacts/frame_store`.

## 4. Sync dữ liệu từ S3 về local

### 4.1. Chuẩn bị AWS CLI

Cài AWS CLI v2 và cấu hình credential bằng AWS profile hoặc IAM role. Không
đặt access key vào Git, root `.env`, `frontend/.env` hoặc command frontend vì
biến `REACT_APP_*` sẽ bị đóng gói vào browser bundle.

Linux/macOS:

```bash
aws configure --profile hcmai
export AWS_PROFILE=hcmai
export AWS_REGION=ap-east-1
aws sts get-caller-identity
```

Windows PowerShell:

```powershell
aws configure --profile hcmai
$env:AWS_PROFILE = "hcmai"
$env:AWS_REGION = "ap-east-1"
aws sts get-caller-identity
```

Mặc định profile hiện tại dùng bucket và region trong
`configs/prepare.yaml` (`storage.s3`). Có thể override bằng biến môi trường:

```text
HCMAI_S3_BUCKET
HCMAI_S3_REGION
HCMAI_S3_ENDPOINT_URL
```

### 4.2. Sync corpus/artifact S3 → local

`aws s3 sync` có thể chạy lại an toàn; file local đã có cùng kích thước sẽ
được giữ lại. Chạy từ root repository.

Linux/macOS/WSL/Git Bash:

```bash
export S3_BUCKET="${HCMAI_S3_BUCKET:-mlecdanbgold-hcmai-hk}"
export S3_REGION="${AWS_REGION:-ap-east-1}"

# BTC keyframes
aws s3 sync "s3://${S3_BUCKET}/data/keyframes/" \
  data/keyframes/ --region "${S3_REGION}" --only-show-errors

# BTC mapping. Prefix trên S3 là map-keyframes, local dùng map_keyframes.
aws s3 sync "s3://${S3_BUCKET}/data/features/map-keyframes/" \
  data/map_keyframes/ --region "${S3_REGION}" --only-show-errors

# Canonical FrameStore
aws s3 sync "s3://${S3_BUCKET}/data/artifacts/frame_store/" \
  artifacts/frame_store/ --region "${S3_REGION}" --only-show-errors

# Enrichment đã publish; chỉ sync những prefix cần dùng.
aws s3 sync "s3://${S3_BUCKET}/data/artifacts/enrichment/" \
  artifacts/enrichment/ --region "${S3_REGION}" --only-show-errors

# Nếu cần lấy retrieval bundle đã publish, xem mục 4.3. Bundle có layout
# versions/<bundle-id>/ nên không sync thẳng cả prefix vào index root.
```

Nếu cần chạy lại BTC ingestion từ dữ liệu nguồn, sync thêm các prefix tương
ứng với layout bucket của team, thường là:

```bash
aws s3 sync "s3://${S3_BUCKET}/data/metadata/" \
  data/metadata/ --region "${S3_REGION}" --only-show-errors
aws s3 sync "s3://${S3_BUCKET}/data/objects/" \
  data/objects/ --region "${S3_REGION}" --only-show-errors
```

Video gốc không cần cho KIS search nếu keyframes và index đã có. Chỉ sync
`data/videos/` khi chạy ASR hoặc một pipeline cần đọc video nguồn.

Kiểm tra sau khi sync:

```bash
test -s artifacts/frame_store/frames.parquet
test -d data/keyframes
test -d artifacts/indexes/visual
find data/keyframes -type f | wc -l
```

PowerShell tương đương:

```powershell
$S3_BUCKET = if ($env:HCMAI_S3_BUCKET) { $env:HCMAI_S3_BUCKET } else { "mlecdanbgold-hcmai-hk" }
$S3_REGION = if ($env:AWS_REGION) { $env:AWS_REGION } else { "ap-east-1" }

aws s3 sync "s3://$S3_BUCKET/data/keyframes/" "data/keyframes/" --region $S3_REGION --only-show-errors
aws s3 sync "s3://$S3_BUCKET/data/features/map-keyframes/" "data/map_keyframes/" --region $S3_REGION --only-show-errors
aws s3 sync "s3://$S3_BUCKET/data/artifacts/frame_store/" "artifacts/frame_store/" --region $S3_REGION --only-show-errors
aws s3 sync "s3://$S3_BUCKET/data/artifacts/enrichment/" "artifacts/enrichment/" --region $S3_REGION --only-show-errors

Test-Path artifacts/frame_store/frames.parquet
Get-ChildItem data/keyframes -Recurse -File | Measure-Object | Select-Object -ExpandProperty Count
```

Nếu layout bucket khác, dùng `aws s3 ls s3://$S3_BUCKET/data/ --recursive`
để tìm prefix trước khi sync. Không dùng `--delete` khi chưa chắc mapping
local/remote; lệnh này có thể xóa file local không tồn tại trên S3.

### 4.3. S3-first index build trên ThunderCompute

Khi muốn để GPU VM tự download input, build index và publish bundle về S3,
dùng entrypoint sau trên máy ThunderCompute. `--s3-dry-run` chỉ inventory,
không download và không build:

```bash
INDEX_DATASET_ARGS=(
  --version btc-keyframes-v1
  --source btc_keyframes
  --frame-store-id btc-keyframes-v1
  --data-root data
  --frames artifacts/frame_store/frames.parquet
  --frame-store-output artifacts/frame_store
  --frame-manifest artifacts/frame_store/manifest.json
  --keyframes-root data/keyframes
  --map-keyframes-root data/map_keyframes
  --context artifacts/corpus/context.parquet
  --transcripts artifacts/enrichment/transcripts
  --expected-video-count 873
  --expected-frame-count 177321
)

PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --s3 --s3-dry-run \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml \
  --s3-config configs/prepare.yaml \
  "${INDEX_DATASET_ARGS[@]}"

PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --s3 --stage all \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml \
  --s3-config configs/prepare.yaml \
  --s3-sync-workers 8 \
  --s3-upload-workers 8 \
  "${INDEX_DATASET_ARGS[@]}"
```

S3 mode chỉ tải keyframes, map_keyframes, FrameStore, FrameContext và
transcript; không tải raw video hoặc các artifact không cần cho index. Chỉ
bundle có `build_report.json` với `status=passed` mới được publish và cập nhật
`latest.json`.

Để lấy bundle mới nhất về local, tải pointer trước rồi sync đúng version được
pointer trỏ tới:

```bash
aws s3 cp "s3://${S3_BUCKET}/data/artifacts/indexes/latest.json" \
  artifacts/indexes/latest.json --region "${S3_REGION}"
VERSION_PREFIX="$(aic/bin/python -c \
  'import json; print(json.load(open("artifacts/indexes/latest.json"))["version_prefix"])')"

for INDEX_NAME in visual context asr_segments; do
  aws s3 sync "s3://${S3_BUCKET}/${VERSION_PREFIX}/${INDEX_NAME}/" \
    "artifacts/indexes/${INDEX_NAME}/" \
    --region "${S3_REGION}" --only-show-errors
done
aws s3 cp "s3://${S3_BUCKET}/${VERSION_PREFIX}/build_report.json" \
  artifacts/indexes/build_report.json --region "${S3_REGION}"
```

PowerShell:

```powershell
aws s3 cp "s3://$S3_BUCKET/data/artifacts/indexes/latest.json" `
  artifacts/indexes/latest.json --region $S3_REGION
$latest = Get-Content artifacts/indexes/latest.json | ConvertFrom-Json
$versionPrefix = $latest.version_prefix
foreach ($indexName in @("visual", "context", "asr_segments")) {
  aws s3 sync "s3://$S3_BUCKET/$versionPrefix/$indexName/" `
    "artifacts/indexes/$indexName/" --region $S3_REGION --only-show-errors
}
aws s3 cp "s3://$S3_BUCKET/$versionPrefix/build_report.json" `
  artifacts/indexes/build_report.json --region $S3_REGION
```

### 4.4. Kiểm tra bundle đã tải từ S3

S3 là đường truyền artifact được hỗ trợ cho workflow ThunderCompute. Giữ
`build_report.json` cùng đúng ba bundle mà pointer `latest.json` chỉ tới; không
trộn các thư mục từ nhiều version. Nếu checkout local có đầy đủ FrameStore,
map keyframe, FrameContext và transcript nguồn, chạy validator trước khi bật
bundle:

```bash
PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --stage validate \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml \
  "${INDEX_DATASET_ARGS[@]}"
```

Checkout chỉ phục vụ bundle không thể chạy lại source-dependent validator;
trong trường hợp đó phải giữ report đã publish và để startup loader kiểm tra
checksum cùng model contract. Xem quy trình đầy đủ tại
[`docs/runbooks/thundercompute-index-build.md`](docs/runbooks/thundercompute-index-build.md).

## 5. Setup backend

### 5.1. Linux

Yêu cầu: Python 3.11+, Git, và nếu dùng S3 thì AWS CLI. Chạy từ root repo:

```bash
python3 --version
python3 -m venv aic
aic/bin/python -m pip install --upgrade pip
aic/bin/python -m pip install -e ".[embedding,reranking,dev]"
```

Tạo root `.env` cho backend. Runtime tự load file này; biến đã export từ môi
trường triển khai vẫn có độ ưu tiên cao hơn:

```bash
cp .env.example .env
```

Khởi động backend:

```bash
PYTHONPATH=.:src aic/bin/python -m uvicorn hcmai.app:app \
  --host 127.0.0.1 --port 8000 --reload
```

Kiểm tra:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8000/api/v1/keyframes/L28_V021_keyframe_000343
```

Frontend dựng URL keyframe từ canonical `frame_id` và `REACT_APP_API_BASE_URL`.
Nếu health báo `remote inference` unavailable, kiểm tra GPU VM/tunnel và
endpoint `/ready` của inference service.

### 5.2. Windows PowerShell

Yêu cầu: Python 3.11+, Node.js LTS, Git, AWS CLI nếu dùng S3. PowerShell
không dùng cú pháp `aic/bin/python`; dùng `aic\Scripts\python.exe`.

```powershell
py --version
py -3.11 -m venv aic
& .\aic\Scripts\python.exe -m pip install --upgrade pip
& .\aic\Scripts\python.exe -m pip install -e ".[embedding,reranking,dev]"
```

Thiết lập biến môi trường cho terminal hiện tại:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$env:HCMAI_DATASET_ROOT = (Join-Path (Get-Location) "data")
$env:HCMAI_METADATA_PATH = (Join-Path (Get-Location) "artifacts\frame_store\frames.parquet")
$env:HCMAI_INDEX_PATH = (Join-Path (Get-Location) "artifacts\indexes\visual")
$env:HCMAI_INFERENCE_BASE_URL = "https://api.iamphuckhang.dev"
```

Khởi động và kiểm tra backend:

```powershell
& .\aic\Scripts\python.exe -m uvicorn hcmai.app:app `
  --host 127.0.0.1 --port 8000 --reload
```

Mở PowerShell khác:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest -Method Get `
  http://127.0.0.1:8000/api/v1/keyframes/L28_V021_keyframe_000343
```

Các biến `$env:*` chỉ tồn tại trong terminal hiện tại. Nếu muốn lưu lâu dài,
dùng `setx` hoặc cấu hình trong profile PowerShell; không lưu Cloudflare secret
vào repository.

### 5.3. Chạy backend bằng local inference

Nếu model API chạy trên chính máy local:

```text
HCMAI_INFERENCE_BASE_URL=http://127.0.0.1:8100
```

Nếu dùng ThunderCompute, frontend/backend vẫn chạy local; chỉ inference URL
trỏ tới hostname Cloudflare. Quy trình GPU là manual: tạo VM, copy source/config
inference đã chọn, kết nối vào VM và chạy service. Repository không còn cung
cấp launcher, cleanup script, hoặc bootstrap template.

```bash
# Tạo instance bằng tnr, chờ trạng thái RUNNING, rồi lấy INSTANCE_ID.
tnr create --gpu l40 --num-gpus 1 --vcpus 8 --template base --disk 200 --yes
INSTANCE_ID=<instance-id>
tnr status --no-wait --json

# Upload source/config inference cho revision sẽ deploy, rồi kết nối vào VM.
# Dùng recursive-copy option của tnr CLI nếu source được đóng gói dưới dạng thư mục.
tnr scp <local-inference-package> "${INSTANCE_ID}:/home/ubuntu/hcmai/" --yes
tnr connect "${INSTANCE_ID}"

# Trên VM: cài pinned dependencies của package và chạy shared inference API.
cd /home/ubuntu/hcmai
HCMAI_LLM_CONFIG=thundercompute/config.yaml \
PYTHONPATH=.:src python -m uvicorn thundercompute.server.api:app \
  --host 127.0.0.1 --port 8100 --workers 1

# Khi xong, xóa instance để tránh phát sinh chi phí.
tnr delete --yes "${INSTANCE_ID}"
```

Giữ Thunder/Cloudflare credential trong CLI profile hoặc secret store riêng
của operator/VM; không commit hoặc truyền token qua command line. Private
operator scripts và `.secrets/` vẫn bị Git ignore, nhưng không có helper
tracked nào để tạo hoặc chạy chúng.

## 6. Setup frontend

Frontend dùng Create React App và mặc định chạy ở `http://127.0.0.1:3000`.
Backend mặc định chạy ở `http://127.0.0.1:8000`.

### Linux/macOS/WSL

```bash
cd frontend
npm ci
cp .env.example .env

# Mở .env và đặt REACT_APP_API_BASE_URL=http://127.0.0.1:8000.
# Giữ lại bucket/region nếu cần video preview từ S3.
npm start
```

### Windows PowerShell

```powershell
Set-Location frontend
npm ci
Copy-Item .env.example .env -Force
(Get-Content .env) -replace '^REACT_APP_API_BASE_URL=.*$', 'REACT_APP_API_BASE_URL=http://127.0.0.1:8000' | Set-Content .env
npm start
```

Mở `http://127.0.0.1:3000`. Nếu backend chạy trên host khác, đặt
`REACT_APP_API_BASE_URL` thành URL đó và thêm origin frontend vào
`HCMAI_CORS_ORIGINS` của backend.

Không đặt `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` hoặc credential tương tự
vào `REACT_APP_*`: React sẽ đưa chúng vào bundle public. Video preview nên dùng
URL tạm thời do backend cấp; keyframe search hiện được phục vụ qua FastAPI.

## 7. Chạy backend + frontend thủ công

Tạo root `.env` từ `.env.example` nếu chưa có. Backend gọi trực tiếp
`HCMAI_INFERENCE_BASE_URL`.

Mở terminal thứ nhất tại repository root:

```bash
[ -f .env ] || cp .env.example .env
PYTHONPATH=.:src aic/bin/python -m uvicorn hcmai.app:app \
  --host 127.0.0.1 --port 8000 --reload
```

Mở terminal thứ hai để chạy frontend:

```bash
npm --prefix frontend ci
cp frontend/.env.example frontend/.env
npm --prefix frontend start
```

Đặt `REACT_APP_API_BASE_URL=http://127.0.0.1:8000` trong `frontend/.env`, rồi
mở `http://127.0.0.1:3000`. Kiểm tra backend bằng:

```bash
curl -sS http://127.0.0.1:8000/health
```

## 8. Kiểm tra và phát triển

Backend tests:

```bash
aic/bin/python -m pytest -q
```

Frontend tests và production build:

```bash
npm --prefix frontend test -- --watchAll=false --runInBand
npm --prefix frontend run build
```

Repository hiện không có release-wrapper script. Chạy trực tiếp các gate được
duy trì và xử lý mọi exit code khác `0` trước khi phát hành:

```bash
PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai thundercompute
PYTHONPATH=.:src aic/bin/python -m pytest -q
CI=true npm --prefix frontend test -- --watchAll=false --runInBand
npm --prefix frontend run build
git diff --check
```

Các test dùng fixture cục bộ; những lệnh trên không gọi remote inference và
không rebuild corpus thật.

## 9. Xử lý lỗi thường gặp

### `remote inference failed (connection)`

Backend đang trỏ về `127.0.0.1:8100` nhưng GPU service chưa chạy trên cùng máy,
hoặc endpoint Cloudflare/Access sai. Kiểm tra:

```bash
curl -i "$HCMAI_INFERENCE_BASE_URL/ready"
```

Sau khi sửa biến môi trường phải restart uvicorn.

### `/api/v1/keyframes/...` trả `404`

Kiểm tra `HCMAI_DATASET_ROOT`:

```bash
export HCMAI_DATASET_ROOT="$PWD/data"
```

Canonical image phải tồn tại dưới `HCMAI_DATASET_ROOT` và `image_path` trong
FrameStore phải resolve đến file đó.

### `reranker.batch_size` vượt quá 16

Giới hạn contract hiện tại của hosted reranker là 16. Đặt
`reranker.batch_size: 16` hoặc thấp hơn trong `thundercompute/config.yaml`, rồi restart
backend.

### Frontend báo không kết nối backend

1. Backend đã listen port 8000 chưa?
2. `REACT_APP_API_BASE_URL` có đúng không?
3. Backend có cho phép origin `http://127.0.0.1:3000`/`http://localhost:3000`
   trong `HCMAI_CORS_ORIGINS` không?
4. Nếu vừa sửa frontend `.env`, stop và chạy lại `npm start`.

## 10. Tài liệu liên quan

- [`AGENTS.md`](AGENTS.md): nguyên tắc làm việc và invariant của repository.
- [`scripts/README.md`](scripts/README.md): các CLI data/enrichment/index.
- [`docs/runbooks/thundercompute-index-build.md`](docs/runbooks/thundercompute-index-build.md):
  build index và đồng bộ ThunderCompute.
- [`thundercompute/README.md`](thundercompute/README.md):
  flow triển khai thủ công create/scp/SSH/delete và inference contracts.
- [`configs/baseline.yaml`](configs/baseline.yaml): cấu hình serving/search.
- [`configs/prepare.yaml`](configs/prepare.yaml): stage policies, S3 transport và model pins; dataset inputs truyền qua CLI.
- [`thundercompute/config.yaml`](thundercompute/config.yaml): model checkpoint và reranker config.
