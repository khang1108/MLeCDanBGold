# Common utilities

The utilities in this package are small, reusable helpers for file I/O,
image loading, timing, and application logging. They do not load models or
large datasets at import time.

## Dependencies

Install the optional libraries used by the helpers when needed:

```bash
aic/bin/python -m pip install -e .
```

- `pyyaml` is used for YAML files.
- `pandas` and a Parquet engine such as `pyarrow` are used for Parquet files.
- `pillow` is used by `load_image`.

## File I/O

`io.py` provides JSON, YAML, and Parquet helpers. Output helpers create parent
directories automatically.

```python
from pathlib import Path

from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    read_parquet,
    read_yaml,
    write_json,
    write_parquet,
    write_yaml,
)

config_path = Path("runs/demo/config.yaml")
write_yaml({"profile": "accurate"}, config_path)
config = read_yaml(config_path)

metrics_path = Path("runs/demo/metrics.json")
write_json({"mrr": 0.82}, metrics_path)
metrics = read_json(metrics_path)

atomic_write(
    "runs/demo/manifest.json",
    lambda temporary: write_json({"status": "complete"}, temporary),
)
```

Parquet helpers use pandas-compatible tables:

```python
import pandas as pd

frames = pd.DataFrame(
    [{"frame_id": "frame-001", "video_id": "video-001"}]
)
write_parquet(frames, "data/metadata/frames.parquet")
loaded_frames = read_parquet("data/metadata/frames.parquet")
```

Additional keyword arguments are forwarded to PyYAML, pandas, or the table's
`to_parquet` method as appropriate.

## Image loading

`load_image` returns a fully loaded, detached Pillow image. The source file can
therefore be safely closed after the function returns. Use `mode` to convert
the image, for example to RGB:

```python
from hcmai.common.utils.image import load_image

image = load_image("data/frames/frame-001.jpg", mode="RGB")
print(image.size)
```

## Timing

Use `Timer` for a measured block. Durations are reported in milliseconds and
use a monotonic high-resolution clock:

```python
from hcmai.common.utils.timing import Timer

with Timer() as timer:
    candidates = retrieve_candidates(query)

print(f"retrieval took {timer.elapsed_ms:.2f} ms")
```

For manually managed timestamps, use `elapsed_ms`:

```python
from time import perf_counter

from hcmai.common.utils.timing import elapsed_ms

started_at = perf_counter()
run_pipeline()
duration_ms = elapsed_ms(started_at)
```

## Logging

`logging.py` wraps the standard library without configuring logging during
import. Configure logging once in an application entry point, then create
named loggers in modules:

```python
from hcmai.common.utils.logging import configure_logging, get_logger

configure_logging("INFO", log_file="runs/demo/pipeline.log")
logger = get_logger(__name__)
logger.info("Pipeline started")
```

Set `force=True` when a script must replace handlers configured earlier in the
process. Prefer `get_logger(__name__)` in reusable modules so log messages keep
their module name.
