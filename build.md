# Chạy tiếp trên 3 máy

Ba máy đã cài sẵn từ trước. Không clone lại, không `aws configure` lại — chỉ vào đúng thư mục cũ rồi copy khối của mình.

Máy 0 giữ `Videos_L21_a`, Máy 1 giữ `Videos_L22_a`. Hai archive đó đã giải nén sẵn trên chính máy đó nên phải để đúng máy đó chạy tiếp; máy khác cầm sang sẽ không có `archive_manifest.json` để nạp lại.

Batch nào đã commit thì `_process_one_batch` tự bỏ qua, cứ chạy đè lên là được.

Trước khi copy, xem thực tế đã commit tới đâu:

```bash
aws s3 ls --recursive s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ | grep _SUCCESS.json
```

## Máy 0 — L21_a, rồi L23_a L24_a L25_a L26_a

```bash
tmux new -s aic

cd MLeCDanBGold
source aic/bin/activate
git pull
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/batches/ runs/custom-raw1fps-v1/state/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/videos/ runs/custom-raw1fps-v1/state/videos/
ZIP_OFFSET=0 ZIP_LIMIT=1 SKIP_APT=1 SKIP_BUILD=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m0-a0.log
ZIP_OFFSET=2 ZIP_LIMIT=4 ALLOW_OFFSET_GAP=1 SKIP_APT=1 SKIP_BUILD=1 SKIP_INFERENCE_SERVER=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m0-a2.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 1 — L22_a, rồi L26_b L26_c L26_d L26_e

```bash
tmux new -s aic

cd MLeCDanBGold
source aic/bin/activate
git pull
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/batches/ runs/custom-raw1fps-v1/state/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/videos/ runs/custom-raw1fps-v1/state/videos/
ZIP_OFFSET=1 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 SKIP_APT=1 SKIP_BUILD=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m1-a1.log
ZIP_OFFSET=6 ZIP_LIMIT=4 ALLOW_OFFSET_GAP=1 SKIP_APT=1 SKIP_BUILD=1 SKIP_INFERENCE_SERVER=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m1-a6.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 2 — L27_a L28_a L29_a L30_a

```bash
tmux new -s aic

cd MLeCDanBGold
source aic/bin/activate
git pull
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/batches/ runs/custom-raw1fps-v1/state/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/videos/ runs/custom-raw1fps-v1/state/videos/
ZIP_OFFSET=10 ZIP_LIMIT=4 ALLOW_OFFSET_GAP=1 SKIP_APT=1 SKIP_BUILD=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m2-a10.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Bảng vị trí archive

| offset | archive | máy |
|---|---|---|
| 0 | Videos_L21_a | 0 |
| 1 | Videos_L22_a | 1 |
| 2 | Videos_L23_a | 0 |
| 3 | Videos_L24_a | 0 |
| 4 | Videos_L25_a | 0 |
| 5 | Videos_L26_a | 0 |
| 6 | Videos_L26_b | 1 |
| 7 | Videos_L26_c | 1 |
| 8 | Videos_L26_d | 1 |
| 9 | Videos_L26_e | 1 |
| 10 | Videos_L27_a | 2 |
| 11 | Videos_L28_a | 2 |
| 12 | Videos_L29_a | 2 |
| 13 | Videos_L30_a | 2 |

## Theo dõi

```bash
grep -ho '"stage": "[a-z_]*"' runs/custom-raw1fps-v1/state/videos/*.json | sort | uniq -c
ls artifacts/custom-raw1fps-v1/batches/*/*/_SUCCESS.json | wc -l
```

## Gộp và finalize

Chạy trên đúng một máy, sau khi cả ba máy đã sync xong.

```bash
aws s3 sync s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ artifacts/custom-raw1fps-v1/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/ runs/custom-raw1fps-v1/state/
rm -f runs/custom-raw1fps-v1/state/*_V*.json
TRANSCRIPTS_ROOT=runs/transcripts-all ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/finalize.log
ls artifacts/custom-raw1fps-v1/batches/*/*/_SUCCESS.json | wc -l
```

## Kiểm tra kết quả

```bash
python -c "
import glob, pandas as pd
root = 'artifacts/custom-raw1fps-v1/batches/*/*/videos/*'
def load(name):
    f = sorted(glob.glob(f'{root}/{name}.parquet'))
    print(name, 'shards:', len(f))
    return pd.concat(map(pd.read_parquet, f), ignore_index=True) if f else pd.DataFrame()
det = load('object_detections')
print(det.label.value_counts().head(30))
print('detected frames:', det.frame_id.nunique())
cap = load('caption')
print('captions:', cap.status.value_counts().to_dict())
print(cap.text.dropna().head(3).tolist())
ocr = load('ocr_regions')
print('ocr regions:', len(ocr))
ctx = load('context')
print('context empty:', ctx.context_text.fillna('').eq('').sum(), '/', len(ctx))
"
```
