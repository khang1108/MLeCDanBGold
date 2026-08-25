# Public C++ contracts

This directory is the public include root for the native C++17 keyframe
extractor. CMake exposes it through the `keyframes_core` target, so source and
test code include headers by their stable project path:

```cpp
#include "hcmai/keyframes_extraction/timestamp_sampler.hpp"
```

Headers define value contracts and small, testable APIs. Their implementations
belong in the sibling [`src/`](../src/README.md) directory. The focused
`extractor.hpp` orchestration contract is the one exception that exposes a
bounded run request for the CLI and smoke tests; it still does not expose
command-line parsing or downstream Parquet conversion.

## Current contracts

| Header | Owns | Does not own |
| --- | --- | --- |
| `types.hpp` | Shared input, configuration, frame, and manifest value types | Parsing, decoding, and filesystem work |
| `config.hpp` | Loading and validating extraction configuration | Applying the configuration to a run |
| `jsonl.hpp` | Strict native input/frame JSONL reading and writing | Video decode and Python/Parquet materialization |
| `frame_index.hpp` | Competition `frame_idx` calculation and internal `frame_id` construction | Sampling and timestamp discovery |
| `timestamp_sampler.hpp` | Nearest decoded-frame selection for fixed target timestamps | FFmpeg decoding and JPEG writing |
| `ffmpeg.hpp` | FFmpeg video decoding metadata and JPEG encoding contracts | Source download, target sampling, and publication |
| `process.hpp` | Shell-free POSIX argv execution and bounded child output | Download argument selection, retries, and state mutation |
| `state.hpp` | Atomic per-video JSON checkpoints and guarded lifecycle transitions | Media operations, bundle validation, and cleanup |
| `extractor.hpp` | Bounded manifest extraction request and outcome summary | CLI parsing, enrichment, publication, and Parquet materialization |

`ffmpeg.hpp` forward-declares `AVFrame` so callers can use decoded images
without exposing FFmpeg implementation headers through every contract.

## Identity and time invariants

Every frame artifact must retain `video_id`, internal `frame_id`,
competition-facing `frame_idx`, and actual `timestamp_ms`.

- `frame_id` is deterministic and internal:
  `{video_id}_raw1fps_{sample_index:09d}`.
- `sample_index`, decoded ordinal, and image filename are never substitutes for
  `frame_idx`.
- `frame_idx` is calculated from the selected decoded frame's actual timestamp:
  `floor(ceil(avg_fps) * timestamp_ms / 1000)`.
- The sampler's requested target timestamp is audit metadata; the selected
  presentation timestamp is the authoritative artifact timestamp.
- `AVStream::avg_frame_rate` is the FPS authority. Missing, non-positive, or
  non-finite values must be rejected.

These rules protect the coordinate submitted to the competition from being
silently replaced by a keyframe order or decoder position.

## Adding or changing a public contract

1. Search existing headers before introducing another type or abstraction.
2. Add the declaration here and place the implementation in `../src/`.
3. Document the file, public class, and public function with its purpose,
   arguments, return value, and relevant exceptions.
4. Keep FFmpeg ownership behind RAII contracts; do not expose borrowed frame
   memory whose lifetime is unclear.
5. Add a focused native test under `../tests/` and register it in
   [`../CMakeLists.txt`](../CMakeLists.txt).

The target is intentionally small and explicit: a clear value type or function
is preferred over a speculative interface hierarchy.
