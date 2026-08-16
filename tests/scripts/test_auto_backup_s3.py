from pathlib import Path


def test_backup_excludes_cache_staging_and_partial_images() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts/auto_backup_s3.sh"
    ).read_text(encoding="utf-8")

    assert '${WORK_ROOT}/artifacts/' in script
    assert '${WORK_ROOT}/.preparation/' in script
    assert 'aws s3 sync "${WORK_ROOT}/"' not in script
    assert '*/.*.partial/*' in script
    assert "source-cache" not in script
