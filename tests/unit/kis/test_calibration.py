from hcmai.common.schemas import RetrievalCandidate, RetrievalSource
from hcmai.pipelines.kis.calibration import CalibrationCase, calibrate_fusion


def test_calibration_selects_measured_modality_without_rewriting_identity():
    visual = RetrievalCandidate(
        frame_id="visual", source_ranks={RetrievalSource.VISUAL: 1}
    )
    ocr = RetrievalCandidate(
        frame_id="gold", source_ranks={RetrievalSource.OCR: 1}
    )
    case = CalibrationCase((visual, ocr), frozenset({"gold"}))

    result = calibrate_fusion(
        [case],
        [
            {RetrievalSource.VISUAL: 1.0, RetrievalSource.OCR: 0.0},
            {RetrievalSource.VISUAL: 0.0, RetrievalSource.OCR: 1.0},
        ],
    )

    assert result.weights[RetrievalSource.OCR] == 1.0
    assert result.mean_top_k_score == 1.0
