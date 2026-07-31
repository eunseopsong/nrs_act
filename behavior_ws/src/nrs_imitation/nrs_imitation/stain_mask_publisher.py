#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish a live stain mask from an RGB camera stream."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from nrs_imitation.hdf5_recorder_base import image_to_rgb_numpy
from nrs_imitation.hdf5_recorder_single_cam_stain_mask import (
    generate_stain_mask_from_rgb,
    make_stain_mask_overlay,
)


def _reliability_from_str(value: str) -> ReliabilityPolicy:
    s = str(value or "best_effort").strip().lower()
    if s in ("reliable", "rel"):
        return ReliabilityPolicy.RELIABLE
    return ReliabilityPolicy.BEST_EFFORT


def _image_qos(depth: int, reliability: ReliabilityPolicy) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=reliability,
    )


def _mono8_to_image_msg(mask: np.ndarray, stamp=None, frame_id: str = "") -> Image:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise RuntimeError(f"mask must be 2D, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1])
    msg.data = arr.tobytes()
    return msg


def _rgb_to_image_msg(rgb: np.ndarray, stamp=None, frame_id: str = "") -> Image:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"RGB image must be (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()
    return msg


class StainMaskPublisher(Node):
    def __init__(self):
        super().__init__("stain_mask_publisher")

        self.declare_parameter("image_topic", "/realsense/vr/color/image_raw")
        self.declare_parameter("mask_topic", "/inference_single_cam/stain_mask")
        self.declare_parameter("overlay_topic", "/inference_single_cam/stain_mask_overlay")
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("image_qos", "best_effort")
        self.declare_parameter("mask_mode", "rgb_threshold")
        self.declare_parameter("task_roi_center_x", 253)
        self.declare_parameter("task_roi_y_end", 110)
        self.declare_parameter("task_roi_half_width", 12)
        self.declare_parameter("tcp_roi_reference_width", 424)
        self.declare_parameter("tcp_roi_reference_height", 240)
        self.declare_parameter("tcp_roi_center_x", 253)
        self.declare_parameter("tcp_roi_center_y", 120)
        self.declare_parameter("tcp_roi_area_fraction", 0.10)
        self.declare_parameter("stain_dark_thresh", 80)
        self.declare_parameter("reflection_v_thresh", 235)
        self.declare_parameter("reflection_s_thresh", 60)
        self.declare_parameter("stain_min_area", 20)
        self.declare_parameter("stain_morph_kernel", 3)
        self.declare_parameter("overlay_alpha", 0.45)
        self.declare_parameter("log_every_n", 60)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.mask_mode = str(self.get_parameter("mask_mode").value).strip().lower()
        self.task_roi_center_x = int(self.get_parameter("task_roi_center_x").value)
        self.task_roi_y_end = int(self.get_parameter("task_roi_y_end").value)
        self.task_roi_half_width = int(self.get_parameter("task_roi_half_width").value)
        self.tcp_roi_reference_width = int(self.get_parameter("tcp_roi_reference_width").value)
        self.tcp_roi_reference_height = int(self.get_parameter("tcp_roi_reference_height").value)
        self.tcp_roi_center_x = int(self.get_parameter("tcp_roi_center_x").value)
        self.tcp_roi_center_y = int(self.get_parameter("tcp_roi_center_y").value)
        self.tcp_roi_area_fraction = float(self.get_parameter("tcp_roi_area_fraction").value)
        self.stain_dark_thresh = int(self.get_parameter("stain_dark_thresh").value)
        self.reflection_v_thresh = int(self.get_parameter("reflection_v_thresh").value)
        self.reflection_s_thresh = int(self.get_parameter("reflection_s_thresh").value)
        self.stain_min_area = int(self.get_parameter("stain_min_area").value)
        self.stain_morph_kernel = int(self.get_parameter("stain_morph_kernel").value)
        self.overlay_alpha = float(self.get_parameter("overlay_alpha").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))
        if self.mask_mode not in ("rgb_threshold", "task_roi", "tcp_roi"):
            raise ValueError(
                f"mask_mode must be rgb_threshold, task_roi, or tcp_roi, got {self.mask_mode}"
            )
        if self.task_roi_half_width < 0:
            raise ValueError("task_roi_half_width must be non-negative")
        if self.tcp_roi_reference_width <= 0 or self.tcp_roi_reference_height <= 0:
            raise ValueError("tcp_roi reference resolution must be positive")
        if not 0.0 < self.tcp_roi_area_fraction <= 1.0:
            raise ValueError("tcp_roi_area_fraction must be in (0, 1]")

        img_qos = _image_qos(
            depth=1,
            reliability=_reliability_from_str(str(self.get_parameter("image_qos").value)),
        )

        self.pub_mask = self.create_publisher(Image, self.mask_topic, 1)
        self.pub_overlay = None
        if self.publish_overlay:
            self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 1)

        self.create_subscription(Image, self.image_topic, self._on_image, img_qos)
        self._count = 0

        self.get_logger().info(
            "[STAIN-MASK-PUB] "
            f"image_topic={self.image_topic}, mask_topic={self.mask_topic}, "
            f"overlay_topic={self.overlay_topic if self.publish_overlay else '(disabled)'}, "
            f"mode={self.mask_mode}, "
            f"task_roi=(center_x={self.task_roi_center_x}, y_end={self.task_roi_y_end}, "
            f"half_width={self.task_roi_half_width}), "
            f"tcp_roi=(ref={self.tcp_roi_reference_width}x{self.tcp_roi_reference_height}, "
            f"center=({self.tcp_roi_center_x},{self.tcp_roi_center_y}), "
            f"area_fraction={self.tcp_roi_area_fraction:.6f}), "
            f"dark_thresh={self.stain_dark_thresh}, min_area={self.stain_min_area}, "
            f"morph_kernel={self.stain_morph_kernel}"
        )

    def _make_task_roi_mask(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        # Dataset masks were authored at 424x240. Scale the configured ROI if
        # the live stream uses another resolution.
        scale_x = float(width) / 424.0
        scale_y = float(height) / 240.0
        center_x = int(round(self.task_roi_center_x * scale_x))
        half_width = max(0, int(round(self.task_roi_half_width * scale_x)))
        y_end = int(round(self.task_roi_y_end * scale_y))
        x0 = max(0, center_x - half_width)
        x1 = min(width, center_x + half_width + 1)
        y1 = min(height, max(0, y_end))

        mask = np.zeros((height, width), dtype=np.uint8)
        if x1 > x0 and y1 > 0:
            mask[:y1, x0:x1] = 255
        return mask

    def _make_tcp_roi_mask(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        scale_x = float(width) / float(self.tcp_roi_reference_width)
        scale_y = float(height) / float(self.tcp_roi_reference_height)
        center_x = int(round(self.tcp_roi_center_x * scale_x))
        center_y = int(round(self.tcp_roi_center_y * scale_y))

        side = max(
            1,
            int(round(np.sqrt(self.tcp_roi_area_fraction * float(width * height)))),
        )
        side = min(side, width, height)
        x0 = min(max(0, center_x - side // 2), width - side)
        y0 = min(max(0, center_y - side // 2), height - side)
        x1 = x0 + side
        y1 = y0 + side

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return mask

    def _on_image(self, msg: Image):
        try:
            rgb = image_to_rgb_numpy(msg)
            if rgb is None:
                raise RuntimeError(f"unsupported image encoding={msg.encoding}")

            if self.mask_mode == "task_roi":
                mask = self._make_task_roi_mask(rgb)
            elif self.mask_mode == "tcp_roi":
                mask = self._make_tcp_roi_mask(rgb)
            else:
                mask = generate_stain_mask_from_rgb(
                    rgb,
                    stain_dark_thresh=self.stain_dark_thresh,
                    reflection_v_thresh=self.reflection_v_thresh,
                    reflection_s_thresh=self.reflection_s_thresh,
                    stain_min_area=self.stain_min_area,
                    stain_morph_kernel=self.stain_morph_kernel,
                )

            frame_id = msg.header.frame_id or "stain_mask"
            self.pub_mask.publish(_mono8_to_image_msg(mask, stamp=msg.header.stamp, frame_id=frame_id))

            if self.pub_overlay is not None:
                overlay = make_stain_mask_overlay(rgb, mask, alpha=self.overlay_alpha)
                self.pub_overlay.publish(_rgb_to_image_msg(overlay, stamp=msg.header.stamp, frame_id=frame_id))

            self._count += 1
            if self._count <= 3 or (self._count % self.log_every_n == 0):
                coverage = 100.0 * float(np.count_nonzero(mask)) / max(1, int(mask.size))
                self.get_logger().info(
                    f"[STAIN-MASK-PUB] #{self._count} mask_shape={tuple(mask.shape)} "
                    f"coverage={coverage:.2f}%"
                )
        except Exception as e:
            self.get_logger().warn(f"[STAIN-MASK-PUB] failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StainMaskPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
