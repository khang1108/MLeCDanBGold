# ThunderCompute Multimodal Index Build

This runbook builds the competition retrieval indexes on one RTX A6000 and
copies only validated bundles back to the local HCMAI checkout. The workflow
uses BTC keyframes as-is; it does not upload raw videos, extract replacement
frames, or regenerate indexes during serving.

## Prepare the organizer mapping explicitly

`configs/indexing.yaml` requires CSV files directly under
`data/map_keyframes/`. The build command deliberately does not unpack or infer
this organizer-owned mapping. Prepare it locally in one of these explicit ways
before pushing inputs:

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

## Configure both machines

Set infrastructure values yourself; the repository does not guess SSH hosts
or machine-specific paths.

```bash
export HCMAI_LOCAL_ROOT=/absolute/path/to/local/hcmai
export HCMAI_THUNDER_HOST=<your-existing-ssh-alias-or-user-at-host>
export HCMAI_THUNDER_ROOT=/absolute/path/on/thundercompute/hcmai
```

The local root must already contain canonical `frames.parquet` and its
manifest, BTC keyframes and mappings, FrameContext and its manifest,
transcripts, and the two indexing configs. Push only those inputs and the
builder source:

```bash
cd "$HCMAI_LOCAL_ROOT"
bash scripts/sync_thundercompute_indexes.sh push-inputs
```

The transfer script has no `--delete` operation and does not include the raw
video corpus.

## Install the remote build environment

Connect using the configured host and change to the configured remote root.
Export the root again in the remote shell because ordinary SSH sessions do not
forward arbitrary environment variables by default.

```bash
ssh "$HCMAI_THUNDER_HOST"
export HCMAI_THUNDER_ROOT=/absolute/path/on/thundercompute/hcmai
cd "$HCMAI_THUNDER_ROOT"
python -m venv aic
aic/bin/python -m pip install -e '.[embedding]'
source aic/bin/activate
```

If the environment already exists, activate it without reinstalling. Verify
that the pinned model revisions in `configs/indexing.models.yaml` are available
to the remote Hugging Face cache before starting an offline build.

## Run and inspect each stage once

Preflight must pass before GPU work. It requires exactly 873 mapping videos and
177,321 canonical/mapping rows, validates the complete organizer join,
projects every staged keyframe path, checks typed FrameContext lineage, and
loads positive-duration unique transcript segments. Duplicate
`(video_id, frame_idx)` coordinates are reported and retained because
`frame_id` is the internal identity.

```bash
python scripts/build_retrieval_indexes.py --stage preflight --config configs/indexing.yaml --model-config configs/indexing.models.yaml
nvidia-smi
python scripts/build_retrieval_indexes.py --stage visual --config configs/indexing.yaml --model-config configs/indexing.models.yaml
python scripts/build_retrieval_indexes.py --stage context --config configs/indexing.yaml --model-config configs/indexing.models.yaml
python scripts/build_retrieval_indexes.py --stage asr --config configs/indexing.yaml --model-config configs/indexing.models.yaml
python scripts/build_retrieval_indexes.py --stage validate --config configs/indexing.yaml --model-config configs/indexing.models.yaml
```

The stages publish to:

```text
artifacts/indexes/visual/
artifacts/indexes/context/
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
python scripts/build_retrieval_indexes.py --stage all --config configs/indexing.yaml --model-config configs/indexing.models.yaml
```

Exit the remote shell and pull only the validated retrieval bundles:

```bash
exit
cd "$HCMAI_LOCAL_ROOT"
bash scripts/sync_thundercompute_indexes.sh pull-indexes
```

Do not promote the downloaded indexes to serving if the report is absent, the
status is not `passed`, or startup rejects any checksum/model contract.
