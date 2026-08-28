# Chạy tiếp trên 3 instance mới

Mỗi máy copy đúng một khối. Hai chỗ phải nhập tay: `tmux new -s aic` và `aws configure`.

Batch nào đã commit trước đó đều được bỏ qua — mỗi khối kéo `state/batches/` và `state/videos/` chung về trước khi chạy. Cố tình không kéo `state/archives/`: máy cũ đã mất, archive dở phải được tải và giải nén lại từ đầu trên máy mới.

Sau `aws configure` chạy `aws sts get-caller-identity`; sai key thì `aws s3 sync` im lặng không kéo gì và pipeline chết ở bước ASR index.

## Máy 0 — L21_a, L23_a, L24_a, L25_a, L26_a

```bash
sudo apt-get update && sudo apt-get install -y tmux aria2
tmux new -s aic

git clone -b feat/detection https://github.com/khang1108/MLeCDanBGold.git
cd MLeCDanBGold
python -m venv aic
source aic/bin/activate
pip install --upgrade pip awscli
aws configure
aws sts get-caller-identity
aws s3 sync s3://mlecdanbgold-db/artifacts/ artifacts/ --exclude 'custom-raw1fps-v1/batches/*'
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/batches/ runs/custom-raw1fps-v1/state/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/videos/ runs/custom-raw1fps-v1/state/videos/
mkdir -p runs/transcripts-all
cp -rsn "$PWD"/artifacts/enrichment/transcripts/L2[1-4] runs/transcripts-all/
cp -rsn "$PWD"/artifacts/transcripts/L2[4-9] "$PWD"/artifacts/transcripts/L30 runs/transcripts-all/
(while sleep 600; do
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=0 ZIP_LIMIT=1 TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m0-a0.log
ZIP_OFFSET=2 ZIP_LIMIT=4 ALLOW_OFFSET_GAP=1 SKIP_APT=1 SKIP_BUILD=1 SKIP_INFERENCE_SERVER=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m0-a2.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 1 — L22_a, L26_b, L26_c, L26_d, L26_e

```bash
sudo apt-get update && sudo apt-get install -y tmux aria2
tmux new -s aic

git clone -b feat/detection https://github.com/khang1108/MLeCDanBGold.git
cd MLeCDanBGold
python -m venv aic
source aic/bin/activate
pip install --upgrade pip awscli
aws configure
aws sts get-caller-identity
aws s3 sync s3://mlecdanbgold-db/artifacts/ artifacts/ --exclude 'custom-raw1fps-v1/batches/*'
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/batches/ runs/custom-raw1fps-v1/state/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/videos/ runs/custom-raw1fps-v1/state/videos/
mkdir -p runs/transcripts-all
cp -rsn "$PWD"/artifacts/enrichment/transcripts/L2[1-4] runs/transcripts-all/
cp -rsn "$PWD"/artifacts/transcripts/L2[4-9] "$PWD"/artifacts/transcripts/L30 runs/transcripts-all/
(while sleep 600; do
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=1 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m1-a1.log
ZIP_OFFSET=6 ZIP_LIMIT=4 ALLOW_OFFSET_GAP=1 SKIP_APT=1 SKIP_BUILD=1 SKIP_INFERENCE_SERVER=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/m1-a6.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 2 — L27_a, L28_a, L29_a, L30_a

```bash
sudo apt-get update && sudo apt-get install -y tmux aria2
tmux new -s aic

git clone -b feat/detection https://github.com/khang1108/MLeCDanBGold.git
cd MLeCDanBGold
python -m venv aic
source aic/bin/activate
pip install --upgrade pip awscli
aws configure
aws sts get-caller-identity
aws s3 sync s3://mlecdanbgold-db/artifacts/ artifacts/ --exclude 'custom-raw1fps-v1/batches/*'
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/batches/ runs/custom-raw1fps-v1/state/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/videos/ runs/custom-raw1fps-v1/state/videos/
mkdir -p runs/transcripts-all
cp -rsn "$PWD"/artifacts/enrichment/transcripts/L2[1-4] runs/transcripts-all/
cp -rsn "$PWD"/artifacts/transcripts/L2[4-9] "$PWD"/artifacts/transcripts/L30 runs/transcripts-all/
(while sleep 600; do
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=10 ZIP_LIMIT=4 ALLOW_OFFSET_GAP=1 TRANSCRIPTS_ROOT=runs/transcripts-all \
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
