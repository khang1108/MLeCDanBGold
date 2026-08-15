"""Module xây dựng Corpus (Tập dữ liệu).

Đóng vai trò là entry point cho các tác vụ gom nhóm dữ liệu thô và làm giàu thành một dataset hoàn chỉnh.

Các tính năng chính:
1. Khởi tạo pipeline: Cung cấp API để bắt đầu quá trình build corpus.
2. Đóng gói (Publish): Cung cấp hàm export corpus ra S3/đĩa cứng.
3. Cấu hình mặc định: Định nghĩa thông số chuẩn để build bộ dataset thi đấu."""

from hcmai.data.corpus_build.config import (
    PinnedModelConfig,
    PreparationModelPins,
    PreparationStagesConfig,
    S3CorpusPreparationConfig,
)
from hcmai.data.corpus_build.pipeline import (
    DefaultPreparationOperations,
    PreparationOperations,
    PreparationPaths,
    PreparationRun,
    S3CorpusPreparationService,
)

__all__ = [
    "DefaultPreparationOperations",
    "PinnedModelConfig",
    "PreparationModelPins",
    "PreparationOperations",
    "PreparationPaths",
    "PreparationRun",
    "PreparationStagesConfig",
    "S3CorpusPreparationConfig",
    "S3CorpusPreparationService",
]
