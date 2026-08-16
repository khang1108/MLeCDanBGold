from pathlib import Path


def test_end_to_end_delegates_to_corpus_preparation_once() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "src/hcmai/data/run_end_to_end.sh"
    ).read_text(encoding="utf-8")

    assert script.count("thunder_batch_launcher.sh") == 1
    for duplicated in (
        "build_embeddings.py",
        "generate_enrichment.py",
        "generate_ocr_enrichment.py",
        "prepare_transcripts.py",
        "build_caption_index.py",
    ):
        assert duplicated not in script
