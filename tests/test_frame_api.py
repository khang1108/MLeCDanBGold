from __future__ import annotations
import asyncio
from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from hcmai.app import create_app
from hcmai.common.schemas import FrameRecord
from hcmai.search import SearchEngine

def test_frame_asset_is_served_only_from_dataset_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "frames" / "safe.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fake-jpeg")
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"secret")
    class Store:
        _records = ()
        def get(self, frame_id):
            path = "frames/safe.jpg" if frame_id == "safe" else "../outside.jpg"
            return FrameRecord(
                frame_id=frame_id, video_id="v1", frame_idx=0,
                timestamp_ms=0, image_path=path, width=1, height=1,
            )
    monkeypatch.setenv("HCMAI_DATASET_ROOT", str(tmp_path))
    app = create_app(SearchEngine(Store(), object()))
    route = next(route for route in app.routes
                 if getattr(route, "path", "").endswith("/{frame_id}/image"))
    assert isinstance(route, APIRoute)
    response = asyncio.run(route.endpoint("safe"))
    assert Path(response.path).read_bytes() == b"fake-jpeg"
    with pytest.raises(HTTPException) as error:
        asyncio.run(route.endpoint("escape"))
    assert error.value.status_code == 404
