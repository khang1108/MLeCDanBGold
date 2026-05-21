IMG_EXTS = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
VIDEO_EXTS = ["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"]

DEFAULT_METADATA_FILENAME = "frames_metadata.parquet"
DEFAULT_LOG_FILENAME = "run.log"

REQUIRED_FRAME_COLUMNS = {
    "row_id",
    "frame_id",
    "video_id",
    "group_id",
    "image_path",
}