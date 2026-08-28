# Chạy pipeline trên 25 instance

Mỗi máy copy đúng một khối, không máy nào trùng khối. Hai chỗ phải nhập tay: `tmux new -s aic` và `aws configure`.

S3 hiện chưa có batch nào commit, cả 14 archive chạy từ đầu.

Mỗi khối chạy một vòng nền đẩy kết quả lên S3 mỗi 10 phút, nên instance chết bất ngờ chỉ mất tối đa 10 phút. Vòng đó sống theo shell tmux, đóng session là nó dừng.

Vòng nền đẩy ba thứ: batch đã commit, ảnh keyframe (`published/`), và state. Ảnh nặng nhất, khoảng 2–3 GB mỗi máy; không đẩy là mất luôn khi instance bị xoá, và UI sẽ không hiện được frame nào. Thư mục `pipeline/` và `frame_store/` trong mỗi batch bị loại vì đó là scratch — riêng `pipeline/enrichment/*/raw/` có một file JSON cho mỗi frame, gần một triệu file, đẩy lên chỉ tổ chậm.

Sau `aws configure` chạy `aws sts get-caller-identity`; sai key thì `aws s3 sync` im lặng không kéo gì và pipeline chết ở bước ASR index.

Số video mỗi archive đếm từ `artifacts/enrichment/transcripts/`: L21 29, L22 31, L23 25, L24 43, L25 88, L26 498, L27 16, L28 24, L29 23, L30 96. Chia 8 ra số batch. L26 bị cắt thành năm phần a–e nên số batch mỗi phần chưa biết chắc; máy nào có cửa sổ batch nằm ngoài cuối archive sẽ thoát ngay, không hỏng gì.

## Máy 1 — Videos_L21_a, batch 0 trở đi

```bash
sudo apt-get update && sudo apt-get install -y tmux aria2
tmux new -s aic
tmux set -g mouse on

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=0 ZIP_LIMIT=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a0-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 2 — Videos_L22_a, batch 0 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=1 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a1-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 3 — Videos_L23_a, batch 0 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=2 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a2-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 4 — Videos_L24_a, batch 0 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=3 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a3-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 5 — Videos_L25_a, batch 0–5

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=4 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=6 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a4-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 6 — Videos_L25_a, batch 6 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=4 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=6 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a4-b6.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 7 — Videos_L26_a, batch 0–4

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=5 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=5 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a5-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 8 — Videos_L26_a, batch 5–9

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=5 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=5 BATCH_LIMIT=5 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a5-b5.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 9 — Videos_L26_a, batch 10 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=5 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=10 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a5-b10.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 10 — Videos_L26_b, batch 0–4

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=6 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=5 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a6-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 11 — Videos_L26_b, batch 5–9

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=6 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=5 BATCH_LIMIT=5 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a6-b5.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 12 — Videos_L26_b, batch 10 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=6 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=10 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a6-b10.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 13 — Videos_L26_c, batch 0–4

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=7 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=5 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a7-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 14 — Videos_L26_c, batch 5–9

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=7 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=5 BATCH_LIMIT=5 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a7-b5.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 15 — Videos_L26_c, batch 10 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=7 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=10 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a7-b10.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 16 — Videos_L26_d, batch 0–6

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=8 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=7 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a8-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 17 — Videos_L26_d, batch 7 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=8 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=7 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a8-b7.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 18 — Videos_L26_e, batch 0–6

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=9 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=7 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a9-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 19 — Videos_L26_e, batch 7 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=9 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=7 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a9-b7.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 20 — Videos_L27_a, batch 0 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=10 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a10-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 21 — Videos_L28_a, batch 0 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=11 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a11-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 22 — Videos_L29_a, batch 0 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=12 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a12-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 23 — Videos_L30_a, batch 0–3

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=13 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a13-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 24 — Videos_L30_a, batch 4–7

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=13 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a13-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 25 — Videos_L30_a, batch 8 trở đi

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
  aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
  aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
  aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
done) &
ZIP_OFFSET=13 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=8 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a13-b8.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ --exclude '*/pipeline/*' --exclude '*/frame_store/*'
aws s3 sync runs/custom-raw1fps-v1/published/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Bảng phân công

| máy | offset | archive | batch |
|---|---|---|---|
| 1 | 0 | Videos_L21_a | 0+ |
| 2 | 1 | Videos_L22_a | 0+ |
| 3 | 2 | Videos_L23_a | 0+ |
| 4 | 3 | Videos_L24_a | 0+ |
| 5 | 4 | Videos_L25_a | 0–5 |
| 6 | 4 | Videos_L25_a | 6+ |
| 7 | 5 | Videos_L26_a | 0–4 |
| 8 | 5 | Videos_L26_a | 5–9 |
| 9 | 5 | Videos_L26_a | 10+ |
| 10 | 6 | Videos_L26_b | 0–4 |
| 11 | 6 | Videos_L26_b | 5–9 |
| 12 | 6 | Videos_L26_b | 10+ |
| 13 | 7 | Videos_L26_c | 0–4 |
| 14 | 7 | Videos_L26_c | 5–9 |
| 15 | 7 | Videos_L26_c | 10+ |
| 16 | 8 | Videos_L26_d | 0–6 |
| 17 | 8 | Videos_L26_d | 7+ |
| 18 | 9 | Videos_L26_e | 0–6 |
| 19 | 9 | Videos_L26_e | 7+ |
| 20 | 10 | Videos_L27_a | 0+ |
| 21 | 11 | Videos_L28_a | 0+ |
| 22 | 12 | Videos_L29_a | 0+ |
| 23 | 13 | Videos_L30_a | 0–3 |
| 24 | 13 | Videos_L30_a | 4–7 |
| 25 | 13 | Videos_L30_a | 8+ |

## Theo dõi

```bash
grep -ho '"stage": "[a-z_]*"' runs/custom-raw1fps-v1/state/videos/*.json | sort | uniq -c
ls artifacts/custom-raw1fps-v1/batches/*/*/_SUCCESS.json | wc -l
```

## Gộp và finalize

Chạy trên đúng một máy, sau khi cả 25 máy đã sync xong. Máy 20 (L27_a, 2 batch) xong sớm nhất nên dùng máy đó. Chạy ở máy local cũng được — bước này chỉ ghép parquet và dựng FAISS từ vector có sẵn, không nạp model và không cần GPU; cần khoảng 50 GB đĩa trống và 8 GB RAM lúc dựng index context.

Không cần tải ảnh keyframe về để finalize. Ảnh nằm sẵn ở `s3://mlecdanbgold-db/runs/custom-raw1fps-v1/published/`, kéo về sau trên đúng cái máy chạy API.

```bash
aws s3 sync s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ artifacts/custom-raw1fps-v1/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/ runs/custom-raw1fps-v1/state/
ls artifacts/custom-raw1fps-v1/batches/*/*/manifest.json | wc -l
FINALIZE_ONLY=1 SKIP_APT=1 SKIP_BUILD=1 SKIP_INFERENCE_SERVER=1 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/finalize.log
```

Dòng `wc -l` phải ra đúng tổng số batch của cả corpus. `finalize` chỉ kiểm tra 14 archive đã ở `cleaned` và không có video nào bị hai batch cùng nhận — nó **không** phát hiện thiếu batch. Thiếu mà vẫn chạy thì ra corpus khuyết mà không báo lỗi.

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
