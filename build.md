# Chạy pipeline trên 25 instance

Mỗi máy copy đúng một khối, không máy nào trùng khối. Hai chỗ phải nhập tay: `tmux new -s aic` và `aws configure`.

S3 hiện chưa có batch nào commit, cả 14 archive chạy từ đầu.

Mỗi khối chạy một vòng nền đẩy kết quả lên S3 mỗi 10 phút, nên instance chết bất ngờ chỉ mất tối đa 10 phút. Vòng đó sống theo shell tmux, đóng session là nó dừng.

Sau `aws configure` chạy `aws sts get-caller-identity`; sai key thì `aws s3 sync` im lặng không kéo gì và pipeline chết ở bước ASR index.

Mỗi archive khoảng 8 batch, chia đôi ở batch 4. Máy `BATCH_LIMIT=4` làm batch 0–3, máy cùng cặp làm batch 4 trở đi.

## Máy 1 — Videos_L21_a, batch 0–3

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
ZIP_OFFSET=0 ZIP_LIMIT=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a0-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 2 — Videos_L21_a, batch 4 trở đi

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
ZIP_OFFSET=0 ZIP_LIMIT=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a0-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 3 — Videos_L22_a, batch 0–3

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
ZIP_OFFSET=1 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a1-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 4 — Videos_L22_a, batch 4 trở đi

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
ZIP_OFFSET=1 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a1-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 5 — Videos_L23_a, batch 0–3

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
ZIP_OFFSET=2 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a2-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 6 — Videos_L23_a, batch 4 trở đi

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
ZIP_OFFSET=2 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a2-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 7 — Videos_L24_a, batch 0–3

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
ZIP_OFFSET=3 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a3-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 8 — Videos_L24_a, batch 4 trở đi

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
ZIP_OFFSET=3 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a3-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 9 — Videos_L25_a, batch 0–3

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
ZIP_OFFSET=4 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a4-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 10 — Videos_L25_a, batch 4 trở đi

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
ZIP_OFFSET=4 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a4-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 11 — Videos_L26_a, batch 0–3

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
ZIP_OFFSET=5 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a5-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 12 — Videos_L26_a, batch 4 trở đi

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
ZIP_OFFSET=5 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a5-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 13 — Videos_L26_b, batch 0–3

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
ZIP_OFFSET=6 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a6-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 14 — Videos_L26_b, batch 4 trở đi

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
ZIP_OFFSET=6 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a6-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 15 — Videos_L26_c, batch 0–3

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
ZIP_OFFSET=7 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a7-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 16 — Videos_L26_c, batch 4 trở đi

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
ZIP_OFFSET=7 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a7-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 17 — Videos_L26_d, batch 0–3

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
ZIP_OFFSET=8 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a8-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 18 — Videos_L26_d, batch 4 trở đi

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
ZIP_OFFSET=8 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a8-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 19 — Videos_L26_e, batch 0–3

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
ZIP_OFFSET=9 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a9-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 20 — Videos_L26_e, batch 4 trở đi

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
ZIP_OFFSET=9 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a9-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 21 — Videos_L27_a, batch 0–3

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
ZIP_OFFSET=10 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 BATCH_LIMIT=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a10-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 22 — Videos_L27_a, batch 4 trở đi

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
ZIP_OFFSET=10 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=4 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a10-b4.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 23 — Videos_L28_a, batch 0 trở đi

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
ZIP_OFFSET=11 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a11-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 24 — Videos_L29_a, batch 0 trở đi

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
ZIP_OFFSET=12 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a12-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Máy 25 — Videos_L30_a, batch 0 trở đi

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
ZIP_OFFSET=13 ZIP_LIMIT=1 ALLOW_OFFSET_GAP=1 BATCH_OFFSET=0 \
  TRANSCRIPTS_ROOT=runs/transcripts-all \
  ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh 2>&1 | tee runs/a13-b0.log
aws s3 sync artifacts/custom-raw1fps-v1/batches/ s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/
aws s3 sync runs/custom-raw1fps-v1/state/ s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/
```

## Bảng phân công

| máy | offset | archive | batch |
|---|---|---|---|
| 1 | 0 | Videos_L21_a | 0–3 |
| 2 | 0 | Videos_L21_a | 4+ |
| 3 | 1 | Videos_L22_a | 0–3 |
| 4 | 1 | Videos_L22_a | 4+ |
| 5 | 2 | Videos_L23_a | 0–3 |
| 6 | 2 | Videos_L23_a | 4+ |
| 7 | 3 | Videos_L24_a | 0–3 |
| 8 | 3 | Videos_L24_a | 4+ |
| 9 | 4 | Videos_L25_a | 0–3 |
| 10 | 4 | Videos_L25_a | 4+ |
| 11 | 5 | Videos_L26_a | 0–3 |
| 12 | 5 | Videos_L26_a | 4+ |
| 13 | 6 | Videos_L26_b | 0–3 |
| 14 | 6 | Videos_L26_b | 4+ |
| 15 | 7 | Videos_L26_c | 0–3 |
| 16 | 7 | Videos_L26_c | 4+ |
| 17 | 8 | Videos_L26_d | 0–3 |
| 18 | 8 | Videos_L26_d | 4+ |
| 19 | 9 | Videos_L26_e | 0–3 |
| 20 | 9 | Videos_L26_e | 4+ |
| 21 | 10 | Videos_L27_a | 0–3 |
| 22 | 10 | Videos_L27_a | 4+ |
| 23 | 11 | Videos_L28_a | 0+ |
| 24 | 12 | Videos_L29_a | 0+ |
| 25 | 13 | Videos_L30_a | 0+ |

## Theo dõi

```bash
grep -ho '"stage": "[a-z_]*"' runs/custom-raw1fps-v1/state/videos/*.json | sort | uniq -c
ls artifacts/custom-raw1fps-v1/batches/*/*/_SUCCESS.json | wc -l
```

## Gộp và finalize

Chạy trên đúng một máy, sau khi cả 25 máy đã sync xong.

```bash
aws s3 sync s3://mlecdanbgold-db/artifacts/custom-raw1fps-v1/batches/ artifacts/custom-raw1fps-v1/batches/
aws s3 sync s3://mlecdanbgold-db/runs/custom-raw1fps-v1/state/ runs/custom-raw1fps-v1/state/
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
