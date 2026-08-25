# Native implementation modules

This directory implements the `keyframes_core` C++17 library and the
`keyframe_extractor` command-line executable. Public contracts live in the
sibling [`include/`](../include/README.md) directory; tests live in
[`../tests/`](../tests/).

The intended extractor lifecycle is:

```text
input manifest + configuration
  -> acquire local source video                 (planned)
  -> FFmpeg decode
  -> deterministic timestamp sampling
  -> frame_idx + internal frame_id assignment
  -> durable / temporary JPEG encoding
  -> validated JSONL + atomic per-video publish (planned)
```

Only the independently testable building blocks are implemented so far. The
orchestration stages marked as planned must not be represented as a working
end-to-end extraction path until their state and publication contracts exist.

## Module status

| Source file | Status | Responsibility |
| --- | --- | --- |
| `config.cpp` | Implemented | Parse JSON configuration via json-c and validate extractor settings. |
| `jsonl.cpp` | Implemented | Strictly read the video input manifest and read/write native frame JSONL rows. |
| `frame_index.cpp` | Implemented | Calculate the competition coordinate and deterministic internal frame ID. |
| `timestamp_sampler.cpp` | Implemented | Select the nearest monotonic decoded frame for each crossed fixed target. |
| `ffmpeg.cpp` | Implemented | Decode the first video stream with FFmpeg and write scaled JPEG representations. |
| `main.cpp` | Minimal | Exposes only `keyframe_extractor --version` until runtime orchestration is ready. |
| `process.cpp` | Implemented | Execute literal POSIX argv values with bounded stdout/stderr and exit/signal capture. |
| `state.cpp` | Implemented | Atomically persist JSON checkpoints and guard lifecycle/provenance transitions. |
| `extractor.cpp` | Placeholder | Reserved for composing download, decode, sampling, encoding, state, and manifests. |

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
- Future subprocess code must use explicit argument vectors, not a shell
  string. Future state code must publish only validated complete bundles via
  same-directory temporary paths and atomic rename.

## Working on this directory

Implement the smallest complete slice and keep its public declaration,
implementation, CMake registration, and focused test together. Every source
file begins with its module purpose; non-trivial functions and classes explain
their purpose, arguments, return value, and failure conditions.

Build and run the native test suite from the repository root:

```bash
cmake -S src/hcmai/data/cpp/keyframes_extraction \
  -B build/keyframes_extraction -DCMAKE_BUILD_TYPE=Debug
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction --output-on-failure
```

The CMake project exports `build/keyframes_extraction/compile_commands.json`.
Use that file for C++ editor tooling so local include resolution matches the
actual `keyframes_core` build.
