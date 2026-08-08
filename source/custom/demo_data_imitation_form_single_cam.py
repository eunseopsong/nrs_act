#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert hdf5_recorder_single_cam.py merged HDF5 output into compact imitation form.

Output HDF5 layout:
  action/position
  action/force
  observations/position
  observations/force
  observations/images/cam0

The single-camera DINOv3 pipeline stores RGB only. Its fixed TCP interaction
ROI is generated inside the image encoder from calibrated pixel coordinates,
so no per-frame stain_mask dataset is written by default.
"""

from __future__ import annotations

from _imitation_form_converter import DATASETS_ROOT, run_cli


def main() -> None:
    run_cli(
        description="Convert single-camera merged HDF5 into compact imitation-form episodes.",
        default_root=DATASETS_ROOT / "polishing" / "single_cam",
        camera_names=("cam0",),
        default_stain_mask_mode="none",
    )


if __name__ == "__main__":
    main()
