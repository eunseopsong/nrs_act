#!/usr/bin/env python3
"""Service-triggered playback of one recorded HDF5 demonstration episode.

This node deliberately bypasses every learned-policy and FLOW executor path.
It publishes the recorded ``action/position`` trajectory, and optionally
``action/force``, on the original dataset time axis.
"""

from __future__ import annotations

import enum
import math
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger


class PlaybackState(enum.Enum):
    IDLE = "IDLE"
    ALIGN = "ALIGN"
    SETTLE = "SETTLE"
    PLAY = "PLAY"
    DONE = "DONE"
    ABORT = "ABORT"


def load_episode(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate a 6-D pose / 3-D force action trajectory."""
    expanded = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(expanded):
        raise FileNotFoundError(f"episode file does not exist: {expanded}")
    with h5py.File(expanded, "r") as h5:
        if "action/position" not in h5 or "action/force" not in h5:
            raise KeyError(
                "episode must contain action/position and action/force datasets"
            )
        position = np.asarray(h5["action/position"], dtype=np.float64)
        force = np.asarray(h5["action/force"], dtype=np.float64)

    if position.ndim != 2 or position.shape[1] != 6:
        raise ValueError(f"action/position must be (T,6), got {position.shape}")
    if force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(f"action/force must be (T,3), got {force.shape}")
    if position.shape[0] < 2 or position.shape[0] != force.shape[0]:
        raise ValueError(
            f"position/force length mismatch: {position.shape[0]} vs {force.shape[0]}"
        )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(force)):
        raise ValueError("episode action contains NaN or Inf")
    # This controller expects /currentP and /cmdMotion translation in mm.
    if float(np.median(np.linalg.norm(position[:, :3], axis=1))) < 10.0:
        raise ValueError("position appears to be in metres; playback expects millimetres")
    return position, force


def interpolate_action(
    position: np.ndarray, force: np.ndarray, sample_position: float
) -> Tuple[np.ndarray, int, float]:
    """Linearly interpolate one 9-D command on the recorded sample axis."""
    sample_position = float(np.clip(sample_position, 0.0, position.shape[0] - 1.0))
    lo = int(math.floor(sample_position))
    hi = min(lo + 1, position.shape[0] - 1)
    alpha = float(sample_position - lo)
    pose6 = (1.0 - alpha) * position[lo] + alpha * position[hi]
    force3 = (1.0 - alpha) * force[lo] + alpha * force[hi]
    return np.concatenate([pose6, force3]).astype(np.float64), lo, alpha


class EpisodePlayback(Node):
    def __init__(self) -> None:
        super().__init__("polishing_episode_playback")

        default_episode = os.path.expanduser(
            "~/nrs_imitation/datasets/polishing/single_cam/20260801_2233/"
            "imitation_form_tcp_roi_square10pct_action_fxy_zero/episode_0.hdf5"
        )
        self.declare_parameter("episode_file", default_episode)
        self.declare_parameter("dataset_hz", 30.0)
        self.declare_parameter("playback_speed", 1.0)
        self.declare_parameter("control_hz", 125.0)
        self.declare_parameter("pose_topic", "/ur10skku/currentP")
        self.declare_parameter("force_topic", "/ur10skku/currentF")
        self.declare_parameter("cmd_topic", "/ur10skku/cmdMotion")
        self.declare_parameter("start_service", "/polishing_episode_playback/start")
        self.declare_parameter("stop_service", "/polishing_episode_playback/stop")
        self.declare_parameter("reset_service", "/polishing_episode_playback/reset")
        self.declare_parameter("status_topic", "/polishing_episode_playback/status")
        self.declare_parameter(
            "observation_ratio_topic",
            "/polishing_episode_playback/observation_ratio",
        )
        self.declare_parameter("observation_ratio_hz", 2.0)

        self.declare_parameter("align_min_duration_sec", 5.0)
        self.declare_parameter("align_max_xyz_speed_mm_s", 20.0)
        self.declare_parameter("align_max_rot_speed_rad_s", 0.15)
        self.declare_parameter("align_max_distance_mm", 150.0)
        self.declare_parameter("align_position_tolerance_mm", 3.0)
        self.declare_parameter("align_rotation_tolerance_rad", 0.03)
        self.declare_parameter("align_hold_sec", 1.0)
        self.declare_parameter("align_timeout_margin_sec", 15.0)

        self.declare_parameter("use_action_force", False)
        self.declare_parameter("force_xy_cmd_enable", False)
        self.declare_parameter("force_z_nonnegative", True)
        self.declare_parameter("max_command_fz_N", 35.0)
        self.declare_parameter("measured_fz_abort_N", 45.0)
        self.declare_parameter("allow_multiple_cmd_publishers", False)
        self.declare_parameter("max_recorded_xyz_step_mm", 3.0)
        self.declare_parameter("max_recorded_rot_step_rad", 0.03)
        self.declare_parameter("hold_final_pose", True)

        self.episode_file = os.path.abspath(
            os.path.expanduser(str(self.get_parameter("episode_file").value))
        )
        self.dataset_hz = max(1e-6, float(self.get_parameter("dataset_hz").value))
        self.playback_speed = max(1e-6, float(self.get_parameter("playback_speed").value))
        self.control_hz = max(1.0, float(self.get_parameter("control_hz").value))
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.align_min_duration_sec = max(
            0.1, float(self.get_parameter("align_min_duration_sec").value)
        )
        self.align_max_xyz_speed_mm_s = max(
            1e-6, float(self.get_parameter("align_max_xyz_speed_mm_s").value)
        )
        self.align_max_rot_speed_rad_s = max(
            1e-6, float(self.get_parameter("align_max_rot_speed_rad_s").value)
        )
        self.align_max_distance_mm = float(
            self.get_parameter("align_max_distance_mm").value
        )
        self.align_position_tolerance_mm = max(
            0.0, float(self.get_parameter("align_position_tolerance_mm").value)
        )
        self.align_rotation_tolerance_rad = max(
            0.0, float(self.get_parameter("align_rotation_tolerance_rad").value)
        )
        self.align_hold_sec = max(0.0, float(self.get_parameter("align_hold_sec").value))
        self.align_timeout_margin_sec = max(
            1.0, float(self.get_parameter("align_timeout_margin_sec").value)
        )

        self.use_action_force = bool(self.get_parameter("use_action_force").value)
        self.force_xy_cmd_enable = bool(
            self.get_parameter("force_xy_cmd_enable").value
        )
        self.force_z_nonnegative = bool(
            self.get_parameter("force_z_nonnegative").value
        )
        self.max_command_fz_N = float(self.get_parameter("max_command_fz_N").value)
        self.measured_fz_abort_N = float(
            self.get_parameter("measured_fz_abort_N").value
        )
        self.allow_multiple_cmd_publishers = bool(
            self.get_parameter("allow_multiple_cmd_publishers").value
        )
        self.hold_final_pose = bool(self.get_parameter("hold_final_pose").value)

        self.position, self.force = load_episode(self.episode_file)
        xyz_steps = np.linalg.norm(np.diff(self.position[:, :3], axis=0), axis=1)
        rot_steps = np.linalg.norm(np.diff(self.position[:, 3:6], axis=0), axis=1)
        max_xyz_step = float(np.max(xyz_steps))
        max_rot_step = float(np.max(rot_steps))
        allowed_xyz_step = float(self.get_parameter("max_recorded_xyz_step_mm").value)
        allowed_rot_step = float(self.get_parameter("max_recorded_rot_step_rad").value)
        if max_xyz_step > allowed_xyz_step:
            raise RuntimeError(
                f"recorded XYZ step {max_xyz_step:.3f}mm exceeds limit "
                f"{allowed_xyz_step:.3f}mm"
            )
        if max_rot_step > allowed_rot_step:
            raise RuntimeError(
                f"recorded rotation-vector step {max_rot_step:.5f}rad exceeds limit "
                f"{allowed_rot_step:.5f}rad"
            )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Float64MultiArray, self.pose_topic, self._on_pose, qos)
        self.create_subscription(Float64MultiArray, self.force_topic, self._on_force, qos)
        self.cmd_publisher = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.observation_ratio_publisher = self.create_publisher(
            Image, str(self.get_parameter("observation_ratio_topic").value), 10
        )
        self.observation_ratio_period = 1.0 / max(
            0.1, float(self.get_parameter("observation_ratio_hz").value)
        )
        self.create_service(
            Trigger, str(self.get_parameter("start_service").value), self._on_start
        )
        self.create_service(
            Trigger, str(self.get_parameter("stop_service").value), self._on_stop
        )
        self.create_service(
            Trigger, str(self.get_parameter("reset_service").value), self._on_reset
        )

        self._lock = threading.Lock()
        self._pose6: Optional[np.ndarray] = None
        self._force3: Optional[np.ndarray] = None
        self.state = PlaybackState.IDLE
        self._align_from: Optional[np.ndarray] = None
        self._align_start_time = 0.0
        self._align_duration = 0.0
        self._settle_start_time: Optional[float] = None
        self._play_start_time = 0.0
        self._last_command: Optional[np.ndarray] = None
        self._last_status_time = 0.0
        self._last_observation_ratio_time = 0.0
        self._last_progress_frame = -1
        self._current_playback_frame = 0
        self._abort_reason = ""

        self.timer = self.create_timer(1.0 / self.control_hz, self._on_timer)
        duration = (self.position.shape[0] - 1) / self.dataset_hz
        self.get_logger().warn(
            "[PLAYBACK] READY but motion is DISARMED until start service is called"
        )
        self.get_logger().info(
            f"[PLAYBACK] file={self.episode_file} frames={len(self.position)} "
            f"dataset_hz={self.dataset_hz:.3f} duration={duration:.3f}s "
            f"speed={self.playback_speed:.3f}x use_action_force={int(self.use_action_force)}"
        )
        self.get_logger().info(
            f"[PLAYBACK] start_pose={np.array2string(self.position[0], precision=5, separator=', ')} "
            f"max_steps=({max_xyz_step:.3f}mm,{max_rot_step:.5f}rad) "
            f"recorded_fz=[{self.force[:,2].min():.3f},{self.force[:,2].max():.3f}]N"
        )

    def _on_pose(self, message: Float64MultiArray) -> None:
        values = np.asarray(message.data, dtype=np.float64)
        if values.size >= 6 and np.all(np.isfinite(values[:6])):
            with self._lock:
                self._pose6 = values[:6].copy()

    def _on_force(self, message: Float64MultiArray) -> None:
        values = np.asarray(message.data, dtype=np.float64)
        if values.size >= 3 and np.all(np.isfinite(values[:3])):
            with self._lock:
                self._force3 = values[:3].copy()

    def _snapshot(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            pose = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force3 is None else self._force3.copy()
        return pose, force

    def _publish(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64).reshape(9).copy()
        if not self.use_action_force:
            command[6:9] = 0.0
        else:
            if not self.force_xy_cmd_enable:
                command[6:8] = 0.0
            if self.force_z_nonnegative:
                command[8] = max(0.0, float(command[8]))
            if self.max_command_fz_N > 0.0:
                command[8] = min(float(command[8]), self.max_command_fz_N)
        if not np.all(np.isfinite(command)):
            self._abort("refused non-finite command")
            return
        message = Float64MultiArray()
        message.data = command.tolist()
        self.cmd_publisher.publish(message)
        self._last_command = command

    def _hold_current(self) -> None:
        pose, _ = self._snapshot()
        if pose is None:
            return
        command = np.zeros(9, dtype=np.float64)
        command[:6] = pose
        self._publish(command)

    def _set_state(self, state: PlaybackState, detail: str = "") -> None:
        self.state = state
        text = f"{state.value}: {detail}" if detail else state.value
        message = String()
        message.data = text
        self.status_publisher.publish(message)
        self.get_logger().warn(f"[PLAYBACK] state={text}")

    def _publish_observation_ratio_dashboard(self, now: float) -> None:
        """Publish an honest modality/source dashboard for deterministic playback.

        There is no policy forward pass in this node, so position/force/image
        observation attribution is exactly zero. Commands come entirely from
        the recorded action trajectory; live pose/force are used only for
        alignment and safety monitoring.
        """
        if now - self._last_observation_ratio_time < self.observation_ratio_period:
            return
        self._last_observation_ratio_time = now

        height, width = 430, 760
        canvas = np.full((height, width, 3), 246, dtype=np.uint8)
        cv2.putText(
            canvas,
            "COMMAND SOURCE / OBSERVATION RATIO",
            (32, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "EP0 deterministic playback (NO MODEL INFERENCE)",
            (32, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

        rows = [
            ("Position observation", 0.0, (90, 165, 235)),
            ("Force observation", 0.0, (90, 200, 120)),
            ("Image observation", 0.0, (210, 150, 80)),
            ("Recorded action playback", 100.0, (70, 100, 230)),
        ]
        x0, x1 = 270, 700
        for index, (label, ratio, color) in enumerate(rows):
            y = 120 + index * 58
            cv2.putText(
                canvas,
                label,
                (32, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (x0, y), (x1, y + 25), (210, 210, 210), -1)
            fill_x = x0 + int(round((x1 - x0) * ratio / 100.0))
            if fill_x > x0:
                cv2.rectangle(canvas, (x0, y), (fill_x, y + 25), color, -1)
            cv2.rectangle(canvas, (x0, y), (x1, y + 25), (80, 80, 80), 1)
            cv2.putText(
                canvas,
                f"{ratio:5.1f}%",
                (x1 - 68, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

        duration = (self.position.shape[0] - 1) / self.dataset_hz
        frame = int(np.clip(self._current_playback_frame, 0, len(self.position) - 1))
        cv2.putText(
            canvas,
            f"State: {self.state.value}    Frame: {frame}/{len(self.position)-1}    "
            f"Time: {frame/self.dataset_hz:.2f}/{duration:.2f} s",
            (32, 374),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Live currentP/currentF: alignment and safety feedback only",
            (32, 405),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "episode_playback_dashboard"
        message.height = height
        message.width = width
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = width * 3
        message.data = canvas.tobytes()
        self.observation_ratio_publisher.publish(message)

    def _abort(self, reason: str) -> None:
        if self.state == PlaybackState.ABORT:
            return
        self._abort_reason = reason
        self._set_state(PlaybackState.ABORT, reason)
        self._hold_current()

    def _on_start(self, _request: Trigger.Request, response: Trigger.Response):
        if self.state in (PlaybackState.ALIGN, PlaybackState.SETTLE, PlaybackState.PLAY):
            response.success = False
            response.message = f"already active: {self.state.value}"
            return response
        publisher_count = self.count_publishers(self.cmd_topic)
        if not self.allow_multiple_cmd_publishers and publisher_count > 1:
            response.success = False
            response.message = (
                f"refused: {publisher_count} publishers exist on {self.cmd_topic}; "
                "stop inference/other command nodes first"
            )
            return response
        pose, measured_force = self._snapshot()
        if pose is None or measured_force is None:
            response.success = False
            response.message = "waiting for currentP/currentF"
            return response
        if (
            self.measured_fz_abort_N > 0.0
            and abs(float(measured_force[2])) >= self.measured_fz_abort_N
        ):
            response.success = False
            response.message = f"measured |Fz| too high: {measured_force[2]:.2f}N"
            return response

        start_pose = self.position[0]
        xyz_distance = float(np.linalg.norm(start_pose[:3] - pose[:3]))
        rot_distance = float(np.linalg.norm(start_pose[3:6] - pose[3:6]))
        if self.align_max_distance_mm > 0.0 and xyz_distance > self.align_max_distance_mm:
            response.success = False
            response.message = (
                f"start is {xyz_distance:.1f}mm away; limit={self.align_max_distance_mm:.1f}mm"
            )
            return response
        duration = max(
            self.align_min_duration_sec,
            1.5 * xyz_distance / self.align_max_xyz_speed_mm_s,
            1.5 * rot_distance / self.align_max_rot_speed_rad_s,
        )
        self._align_from = pose.copy()
        self._align_start_time = time.monotonic()
        self._align_duration = duration
        self._settle_start_time = None
        self._last_progress_frame = -1
        self._abort_reason = ""
        self._set_state(
            PlaybackState.ALIGN,
            f"distance={xyz_distance:.2f}mm/{rot_distance:.4f}rad duration={duration:.2f}s",
        )
        response.success = True
        response.message = (
            f"alignment started: {xyz_distance:.2f}mm/{rot_distance:.4f}rad, "
            f"duration={duration:.2f}s; playback will follow automatically"
        )
        return response

    def _on_stop(self, _request: Trigger.Request, response: Trigger.Response):
        self._abort("operator stop service")
        response.success = True
        response.message = "playback aborted; holding current pose with zero commanded force"
        return response

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response):
        if self.state in (PlaybackState.ALIGN, PlaybackState.SETTLE, PlaybackState.PLAY):
            response.success = False
            response.message = "stop active playback before reset"
            return response
        self._abort_reason = ""
        self._set_state(PlaybackState.IDLE, "motion disarmed")
        response.success = True
        response.message = "reset to IDLE; call start to arm"
        return response

    def _on_timer(self) -> None:
        pose, measured_force = self._snapshot()
        now = time.monotonic()
        self._publish_observation_ratio_dashboard(now)
        if now - self._last_status_time >= 1.0:
            self._last_status_time = now
            status = String()
            status.data = self.state.value
            self.status_publisher.publish(status)

        if self.state in (PlaybackState.IDLE, PlaybackState.DONE):
            if self.state == PlaybackState.DONE and self.hold_final_pose:
                command = np.zeros(9, dtype=np.float64)
                command[:6] = self.position[-1]
                self._publish(command)
            return
        if self.state == PlaybackState.ABORT:
            self._hold_current()
            return
        if pose is None or measured_force is None:
            self._abort("lost currentP/currentF")
            return
        if (
            self.measured_fz_abort_N > 0.0
            and abs(float(measured_force[2])) >= self.measured_fz_abort_N
        ):
            self._abort(
                f"measured |Fz|={abs(float(measured_force[2])):.2f}N exceeds "
                f"{self.measured_fz_abort_N:.2f}N"
            )
            return

        if self.state == PlaybackState.ALIGN:
            if self._align_from is None:
                self._abort("internal alignment start pose missing")
                return
            elapsed = now - self._align_start_time
            tau = float(np.clip(elapsed / self._align_duration, 0.0, 1.0))
            smooth = 3.0 * tau * tau - 2.0 * tau * tau * tau
            command = np.zeros(9, dtype=np.float64)
            command[:6] = (1.0 - smooth) * self._align_from + smooth * self.position[0]
            self._publish(command)
            if tau >= 1.0:
                self._settle_start_time = None
                self._set_state(PlaybackState.SETTLE, "waiting for measured start pose")
            return

        if self.state == PlaybackState.SETTLE:
            command = np.zeros(9, dtype=np.float64)
            command[:6] = self.position[0]
            self._publish(command)
            pos_error = float(np.linalg.norm(pose[:3] - self.position[0, :3]))
            rot_error = float(np.linalg.norm(pose[3:6] - self.position[0, 3:6]))
            reached = (
                pos_error <= self.align_position_tolerance_mm
                and rot_error <= self.align_rotation_tolerance_rad
            )
            if reached:
                if self._settle_start_time is None:
                    self._settle_start_time = now
                elif now - self._settle_start_time >= self.align_hold_sec:
                    self._play_start_time = now
                    self._set_state(PlaybackState.PLAY, "frame=0")
            else:
                self._settle_start_time = None
            timeout = self._align_duration + self.align_timeout_margin_sec
            if now - self._align_start_time > timeout:
                self._abort(
                    f"alignment timeout: pos_err={pos_error:.2f}mm "
                    f"rot_err={rot_error:.4f}rad"
                )
            return

        if self.state == PlaybackState.PLAY:
            sample_position = (
                (now - self._play_start_time) * self.dataset_hz * self.playback_speed
            )
            command, frame, _alpha = interpolate_action(
                self.position, self.force, sample_position
            )
            self._current_playback_frame = frame
            self._publish(command)
            if frame // 30 != self._last_progress_frame // 30:
                self._last_progress_frame = frame
                self.get_logger().info(
                    f"[PLAYBACK] frame={frame}/{len(self.position)-1} "
                    f"t={frame/self.dataset_hz:.2f}s "
                    f"xyz={np.array2string(command[:3], precision=2, separator=',')} "
                    f"cmd_fz={command[8]:.2f}N meas_fz={measured_force[2]:.2f}N"
                )
            if sample_position >= self.position.shape[0] - 1:
                self._set_state(PlaybackState.DONE, "trajectory complete; force command zero")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodePlayback()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("[PLAYBACK] keyboard interrupt; holding current pose")
        node._hold_current()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
