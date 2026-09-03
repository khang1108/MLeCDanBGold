# ThunderCompute Multimodal Index Build

This runbook builds the competition retrieval indexes on one RTX A6000 and
publishes only validated bundles through S3. The workflow uses BTC keyframes
as-is; it does not upload raw videos, extract replacement frames, or regenerate
indexes during serving.

## Prepare the organizer mapping explicitly

The build command receives all dataset-specific inputs through CLI arguments;
`configs/prepare.yaml` only owns indexing policies, output layout, storage, and
model pins. The organizer mapping is passed explicitly through
`--map-keyframes-root`; the command does not unpack or infer it. Prepare it
locally in one of these explicit ways before publishing the canonical inputs:

```bash
cd /absolute/path/to/local/hcmai
mkdir -p data/map_keyframes

# If the supplied archive is present:
unzip -j data/features/map-keyframes.zip '*.csv' -d data/map_keyframes

# Or, if the organizer maps are already extracted in this checkout:
rsync -a data/features/map-keyframes/ data/map_keyframes/
```

Use only one source and inspect `data/map_keyframes/` before continuing. An
alternative location may be passed explicitly with
`--map-keyframes-root /absolute/path/to/map_keyframes`; never substitute an
image filename, keyframe order, FPS estimate, or timestamp for the BTC map.

Keep this dataset contract in the shell session and append it to every index
stage command. Change it for another corpus; do not add the values back to the
YAML:

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
  --context artifacts/enrichment/context_vi/frame_context_v1.parquet
  --transcripts artifacts/enrichment/transcripts
  --expected-video-count 873
  --expected-frame-count 177321
)
```

## Publish canonical inputs to S3

The source checkout must contain canonical `frames.parquet` and its manifest,
BTC keyframes and mappings, FrameContext and its manifest, transcripts, and the
two indexing configs. Publish those artifacts under the prefixes documented in
the repository README; `configs/prepare.yaml` owns bucket transport under
`storage.s3`. S3 is the supported input and output transfer boundary for this
workflow, and raw videos are not part of the index build transfer.

## S3-first ThunderCompute workflow

Let ThunderCompute stage the published inputs on its local NVMe disk. This
keeps the build restartable. On the ThunderCompute host, configure the normal
AWS credential chain (or attach an instance role) and verify bucket access
without placing credentials in this repository or in shell history:

```bash
cd /absolute/path/on/thundercompute/hcmai
aws s3 ls s3://mlecdanbgold-hcmai-hk/data/keyframes/ --max-items 1
```

The retrieval CLI downloads only the canonical FrameStore, BTC keyframes, BTC
`map_keyframes`, FrameContext, and timestamped transcripts. It does not
download raw videos. The mapping prefix is intentionally different from its
local destination: the organizer archive is published below
`data/features/map-keyframes/`; the local destination is selected through
`--map-keyframes-root`.

Start with an inventory-only check, then run the complete local GPU build and
S3 publication:

```bash
PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --s3 \
  --s3-dry-run \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml \
  --s3-config configs/prepare.yaml \
  "${INDEX_DATASET_ARGS[@]}"

PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --s3 \
  --stage all \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml \
  --s3-config configs/prepare.yaml \
  --s3-sync-workers 8 \
  --s3-upload-workers 8 \
  "${INDEX_DATASET_ARGS[@]}"
```

The `models` section in `configs/prepare.yaml` starts Visual/SigLIP and evidence/BGE at batch
size `128`, suitable as the first RTX A6000 measurement point. S3 mode uses
local model adapters; do not combine it with `--inference-url`. The download
is resumable by byte size and tqdm/log counters report downloaded, skipped, and
failed files. Visual embedding additionally keeps a checkpoint and writes
unreadable-image failures to
`artifacts/indexes/.visual-checkpoints/visual_embedding_failures.json`.

On success, the three indexes and `build_report.json` are uploaded under
`s3://<bucket>/data/artifacts/indexes/versions/<bundle-id>/`; `_SUCCESS.json`
and then `latest.json` are written only after all checksummed artifacts have
uploaded. A failed or interrupted run leaves the previous `latest.json`
untouched. Record the printed version and latest keys for serving or for a
later explicit `aws s3 sync` to another checkout:

```bash
aws s3 cp s3://mlecdanbgold-hcmai-hk/data/artifacts/indexes/latest.json -
```

## Install the remote build environment

Connect using the manual provider workflow documented in
[`llm/README.md`](../../llm/README.md), then install the
build environment in the remote checkout.

```bash
cd /absolute/path/on/thundercompute/hcmai
python -m venv aic
aic/bin/python -m pip install -e '.[embedding]'
source aic/bin/activate
```

If the environment already exists, activate it without reinstalling. Verify
that the pinned model revisions in `configs/prepare.yaml` are available
to the remote Hugging Face cache before starting an offline build.

## Use a private embedding VM from the local build host

When SigLIP and BGE are hosted separately through the private Cloudflare
service, the fast-track builder can keep canonical data and index publication
local while explicitly offloading only embedding inference. The remote service
must advertise the exact pinned model names and revisions from
the `models` section of `configs/prepare.yaml` on `/ready`.

```bash
export HCMAI_INFERENCE_BASE_URL="https://<private-api-hostname>"

PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --stage all \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml \
  --inference-url "$HCMAI_INFERENCE_BASE_URL" \
  "${INDEX_DATASET_ARGS[@]}"
```

`--inference-url` is deliberate: the command never silently reads an endpoint
from the environment. It verifies that requested Visual and BGE capabilities
are loaded and pinned correctly before an embedding stage begins. It sends
SigLIP images to the visual endpoint and both FrameContext and ASR text to the
BGE `text` family. Leave off the option to build with local models instead.

The A6000 starting batch is 128 in both `llm/config.yaml` and
`configs/prepare.yaml`; adjust its `models` section only after observing actual VRAM and
latency with the corpus's real image/text lengths. The builder has one active
embedding batch at a time. Sending several simultaneous requests to one
single-worker GPU service does not provide server-side microbatching and can
increase memory pressure.

## Run and inspect each stage once

Preflight must pass before GPU work. It requires exactly 873 mapping videos and
177,321 canonical/mapping rows, validates the complete organizer join,
projects every staged keyframe path, checks typed FrameContext lineage, and
loads positive-duration unique transcript segments. Duplicate
`(video_id, frame_idx)` coordinates are reported and retained because
`frame_id` is the internal identity.

```bash
python scripts/build_retrieval_indexes.py --stage preflight --config configs/prepare.yaml --model-config configs/prepare.yaml "${INDEX_DATASET_ARGS[@]}"
nvidia-smi
python scripts/build_retrieval_indexes.py --stage visual --config configs/prepare.yaml --model-config configs/prepare.yaml "${INDEX_DATASET_ARGS[@]}"
python scripts/build_retrieval_indexes.py --stage context --config configs/prepare.yaml --model-config configs/prepare.yaml "${INDEX_DATASET_ARGS[@]}"
python scripts/build_retrieval_indexes.py --stage asr --config configs/prepare.yaml --model-config configs/prepare.yaml "${INDEX_DATASET_ARGS[@]}"
python scripts/build_retrieval_indexes.py --stage validate --config configs/prepare.yaml --model-config configs/prepare.yaml "${INDEX_DATASET_ARGS[@]}"
```

The stages publish to:

```text
artifacts/indexes/visual/
artifacts/indexes/context_vi/
artifacts/indexes/asr_segments/
artifacts/indexes/build_report.json
```

`validate` round-trips the v2 checksummed loaders, verifies canonical Visual
coverage, Context subset identity, segment-native ASR identity, and configured
model/revision/dimension/normalization relationships. It writes
`build_report.json` only after every check passes. Inspect that report and
confirm `"status": "passed"` before transfer.

After the individual stages have passed once, the faster single-process option
builds Visual first, releases its GPU model/cache, then reuses one BGE-M3
adapter for Context and ASR:

```bash
python scripts/build_retrieval_indexes.py --stage all --config configs/prepare.yaml --model-config configs/prepare.yaml "${INDEX_DATASET_ARGS[@]}"
```

Visual embedding displays a canonical-frame progress bar. Context and ASR show
their own text progress bars. If a strict Visual build finds unreadable images,
the command writes an atomic failure report at
`artifacts/indexes/.visual-checkpoints/visual_embedding_failures.json` beside
its resumable checkpoints; a later clean repaired run removes that stale report.

After the successful S3 publication, download exactly the immutable version
named by `latest.json` into the serving checkout:

```bash
export S3_BUCKET=mlecdanbgold-hcmai-hk
export S3_REGION="${HCMAI_S3_REGION:-ap-east-1}"
cd /absolute/path/to/serving/hcmai
export HCMAI_LOCAL_ROOT="$PWD"

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

Do not mix directories from different version prefixes. Keep the published
`build_report.json` beside the exact bundles it describes. Startup performs
checksum and model-contract checks as a separate serving guard.

## Promote a validated bundle safely

Promotion is a serving configuration change, not an opportunity to rebuild or
repair artifacts. Confirm the S3 download came from one immutable version and
retain `build_report.json` with the exact bundles it describes. Where the
serving checkout also has the full offline source inputs (especially the
organizer mapping, FrameContext, and transcripts), independently run the
source-dependent validator:

```bash
cd "$HCMAI_LOCAL_ROOT"
aic/bin/python scripts/build_retrieval_indexes.py \
  --stage validate \
  --config configs/prepare.yaml \
  --model-config configs/prepare.yaml
```

When that local validator is applicable, accept a bundle only when it succeeds
and its report has `"status": "passed"`. Confirm the Visual row count is the
complete canonical corpus expected by the active indexing configuration;
Context and ASR counts must match their own usable source rows. A
bundle-only serving checkout cannot rerun this validator without those inputs;
in that case preserve the verified remote report and let startup validate the
checksums and model contracts. The validator and startup loaders check the v2
artifact checksums plus dataset version, model name, immutable revision, vector
dimension, and normalization contracts. A report or bundle from a different
configuration/model revision is not an acceptable substitute.

The standard profile is Visual + optional FrameContext + optional projected
ASR. Point any non-default bundle locations at the three complete directories;
do not point a profile at a partial staging directory:

```bash
export HCMAI_INDEX_PATH="$HCMAI_LOCAL_ROOT/artifacts/indexes/visual"
export HCMAI_CONTEXT_INDEX_PATH="$HCMAI_LOCAL_ROOT/artifacts/indexes/context_vi"
export HCMAI_ASR_SEGMENT_INDEX_PATH="$HCMAI_LOCAL_ROOT/artifacts/indexes/asr_segments"
```

Restart the service using the established deployment command. Startup must
load the required Visual bundle and must not regenerate an index. Check the
service's retrieval observability or a controlled query to confirm the active
sources and canonical frame identities. If Context or ASR is intentionally
absent, record that degraded state explicitly; it is not evidence that the
full B1/B2 profile has been validated.

There is no online specialist-index rollback profile. If a Context or
segment-ASR bundle is missing, startup reports that degraded state and keeps
the canonical Visual path; rebuild or resynchronize the immutable bundle
offline before enabling that modality.

## Unperformed external smoke evidence

This repository checkout currently has no extracted
`data/map_keyframes/` organizer mapping, no published complete index bundles,
and no repository-held B0/B1/B2 manual query set. Consequently, the commands
above document the release procedure but do not constitute a completed
full-corpus validation or qualitative smoke result. Do not infer retrieval
accuracy or production readiness from unit tests or a successful CLI help
command.
