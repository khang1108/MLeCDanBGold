# Native implementation modules

This directory implements the `keyframes_core` C++17 library and the
`keyframe_extractor` command-line executable. Public contracts live in the
sibling [`include/`](../include/README.md) directory; tests live in
[`../tests/`](../tests/).

The intended extractor lifecycle is:

```text
input manifest + configuration
  -> acquire local source video
  -> FFmpeg decode
  -> deterministic timestamp sampling
  -> frame_idx + internal frame_id assignment
  -> durable / temporary JPEG encoding
  -> validated JSONL + atomic per-video staging manifest
  -> enrichment_pending checkpoint
```

The native extraction path is complete through `enrichment_pending`. Later
enrichment, durable publication, cleanup, and corpus materialization remain
separate commands so an offline run cannot silently discard required evidence.

## Module status

| Source file | Status | Responsibility |
| --- | --- | --- |
| `config.cpp` | Implemented | Parse JSON configuration via json-c and validate extractor settings. |
| `jsonl.cpp` | Implemented | Strictly read the video input manifest and read/write native frame JSONL rows. |
| `frame_index.cpp` | Implemented | Calculate the competition coordinate and deterministic internal frame ID. |
| `timestamp_sampler.cpp` | Implemented | Select the nearest monotonic decoded frame for each crossed fixed target. |
| `ffmpeg.cpp` | Implemented | Decode the first video stream with FFmpeg and write scaled JPEG representations. |
| `main.cpp` | Implemented | Parses `extract` arguments, invokes the library, prints JSON summaries, and returns documented exit codes. |
| `process.cpp` | Implemented | Execute literal POSIX argv values with bounded stdout/stderr and exit/signal capture. |
| `state.cpp` | Implemented | Atomically persist JSON checkpoints and guard lifecycle/provenance transitions. |
| `extractor.cpp` | Implemented | Prepare one source, decode/sample/encode frames, validate a staging bundle, write its manifest, and reach `enrichment_pending`. |

## Implementation boundaries

- `ffmpeg.cpp` owns FFmpeg format, codec, packet, frame, and scaling lifetimes
  through RAII. A `DecodedFrame` owns a refcounted clone so it remains valid
  after `VideoDecoder::next()` advances.
- `timestamp_sampler.cpp` selects from actual decoded presentation timestamps;
  it does not invent timestamps or calculate competition coordinates.
- `frame_index.cpp` alone implements
  `floor(ceil(avg_fps) * timestamp_ms / 1000)`. Keep `frame_idx` separate from
  `sample_index` and decode ordinal.
- `jsonl.cpp` is the native artifact boundary only. Python owns later
  validation/materialization into FrameRecord and Parquet.
- `extractor.cpp` invokes yt-dlp only through literal argument vectors. An
  optional local `source_root/{video_id}.mp4` copy keeps smoke tests offline.
- A failed or interrupted attempt removes only that video's staging/source
  artifacts before retry; a retained `enrichment_pending` or later state is
  skipped so Task 7 can safely consume it.
- Native frames JSONL and per-video manifest publication use same-directory
  temporary files and atomic rename after row/image validation.

## Working on this directory

Implement the smallest complete slice and keep its public declaration,
implementation, CMake registration, and focused test together. Every source
file begins with its module purpose; non-trivial functions and classes explain
their purpose, arguments, return value, and failure conditions.

Build and run the native test suite from the repository root:

```bash
cmake -S offline/keyframes/keyframes_extraction \
  -B build/keyframes-extraction -DCMAKE_BUILD_TYPE=Debug
cmake --build build/keyframes-extraction --parallel
ctest --test-dir build/keyframes-extraction --output-on-failure
```

The CMake project exports `build/keyframes-extraction/compile_commands.json`.
Use that file for C++ editor tooling so local include resolution matches the
actual `keyframes_core` build.
