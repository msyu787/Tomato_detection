#!/usr/bin/env python3
"""CLI: YOLO + ByteTrack → ``tracking.pipelines.bytetrack_pipeline.run``."""

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.tracking.pipelines import bytetrack_pipeline

# ---------------------------------------------------------------------------
# 설정 (여기서만 수정)
# ---------------------------------------------------------------------------

CONFIG = {
    "source": "notebook/rgb.mp4",
    "model_path": "runs/yolo26_custom_tomato/trained_yolo26_custom.pt",
    "output_path": "tracking_result/basic_bytetrack.mp4",
    "show_window": True,
    "conf": 0.5,
    "iou": 0.3,
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.3,
    "minimum_consecutive_frames": 1,
    "high_conf_det_threshold": 0.25,
    "frame_rate": 30,
    "motion_compensation": True,
    "motion_max_points": 900,
    "motion_min_distance": 6,
    "motion_block_size": 5,
    "motion_quality_level": 0.003,
    "motion_ransac_reproj_threshold": 2.5,
    "show_trace": False,
    "trace_length": 30,
}


def main() -> None:
    result = bytetrack_pipeline.run(CONFIG)
    print(
        f"[결과] ripe {len(result['unique_ids'][0])}개 / "
        f"unripe {len(result['unique_ids'][1])}개 (누적 고유 ID 기준)"
    )


if __name__ == "__main__":
    main()
