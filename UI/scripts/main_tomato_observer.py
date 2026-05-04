#!/usr/bin/env python3
"""CLI entry for tomato observer tracking pipeline.

Execution example:
  uv run python scripts/main_tomato_observer.py
"""

from __future__ import annotations

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.tracking.pipelines import tomato_observer_pipeline


CONFIG = {
    "tracker_type": "bytetrack",  # "bytetrack" or "sort"
    # "opencv" (default) or "zed" — ZED needs Stereolabs SDK + pyzed; web UI: tomato_observer_app.py.
    "camera_backend": "opencv",
    "source": 1,  # webcam index or video path (opencv only)
    "camera_width": 1280,
    "camera_height": 720,
    "model_path": "models/yolo26n_640.pt",  # fallback: latest models/*.pt
    "device": "0",  # "cpu" or CUDA index
    "show_window": True,
    "output_path": None,  # e.g. "outputs/tracking/tomato_observer.mp4"
    "left_only": True,
    "use_center_roi": True,
    "roi_width_ratio": 0.7,
    "roi_height_ratio": 1.0,
    "conf": 0.5,
    "nms_iou": 0.3,
    "min_box_area": 1000,
    "use_stabilization": False,
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.3,
    "motion_compensation": True,
}

if __name__ == "__main__":
    tomato_observer_pipeline.run(CONFIG)
