#!/usr/bin/env python3
"""Minimal operational FLOW inference with the two requested diagnostics."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ckpt_dir = LaunchConfiguration("ckpt_dir")
    act_root = LaunchConfiguration("act_root")
    policy_class = LaunchConfiguration("policy_class")
    ckpt_auto_subdir = LaunchConfiguration("ckpt_auto_subdir")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    image_topic = LaunchConfiguration("image_topic")
    modality_every_n = LaunchConfiguration("modality_every_n")
    vector_horizon = LaunchConfiguration("vector_horizon")
    metrics_log_enable = LaunchConfiguration("metrics_log_enable")
    metrics_log_dir = LaunchConfiguration("metrics_log_dir")
    metrics_run_tag = LaunchConfiguration("metrics_run_tag")
    use_stain_mask = LaunchConfiguration("use_stain_mask")
    ptp9d_segment_points = LaunchConfiguration("ptp9d_segment_points")
    ptp9d_segment_stride = LaunchConfiguration("ptp9d_segment_stride")
    ptp9d_target_velocity_mm_s = LaunchConfiguration("ptp9d_target_velocity_mm_s")
    inference_mode = LaunchConfiguration("inference_mode")
    track_use_ptp9d_service = PythonExpression(
        ["'true' if '", inference_mode, "' != 'topic_publish' else 'false'"]
    )
    ptp9d_use_stream = PythonExpression(
        ["'true' if '", inference_mode, "' == 'service_stream' else 'false'"]
    )

    base_launch = PathJoinSubstitution(
        [FindPackageShare("nrs_imitation"), "launch", "inference_gradcam_single_cam.launch.py"]
    )
    return LaunchDescription(
        [
            # NOTE: ckpt_dir defaults to a pinned FLOW checkpoint. Switching
            # policy_class to BSPLINE requires passing a matching BSPLINE
            # ckpt_dir too (or "" to auto-select the latest one) -- inference_core
            # raises a clear error instead of silently loading mismatched weights.
            DeclareLaunchArgument(
                "ckpt_dir",
                default_value=(
                    "/home/eunseop/nrs_imitation/checkpoints/flow/polishing/"
                    "single_cam/20260802_1549"
                ),
            ),
            DeclareLaunchArgument("act_root", default_value="/home/eunseop/nrs_imitation"),
            DeclareLaunchArgument("policy_class", default_value="FLOW"),  # FLOW | BSPLINE
            DeclareLaunchArgument("ckpt_auto_subdir", default_value="polishing/single_cam"),
            DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
            DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
            DeclareLaunchArgument(
                "image_topic", default_value="/realsense/vr/color/image_raw"
            ),
            DeclareLaunchArgument("modality_every_n", default_value="1"),
            DeclareLaunchArgument("vector_horizon", default_value="30"),
            # Optional per-run CSV metrics log for offline FLOW-vs-BSPLINE
            # comparison (see scripts/compare_policy_runs.py).
            DeclareLaunchArgument("metrics_log_enable", default_value="false"),
            DeclareLaunchArgument("metrics_log_dir", default_value=""),
            DeclareLaunchArgument("metrics_run_tag", default_value=""),
            # Default "true" matches the pinned legacy FLOW ckpt_dir default
            # above. Newer dinov3/use_tcp_roi checkpoints (both FLOW and
            # BSPLINE) were trained with use_stain_mask=False -- pass
            # use_stain_mask:=false for those or inference_core refuses to
            # start (use_stain_mask mismatch: checkpoint vs inference_arg).
            DeclareLaunchArgument("use_stain_mask", default_value="true"),
            # service_call (default) = TRACK stage drives the robot via
            #   discrete, batched PTP9D service calls -- each call blends
            #   ptp9d_segment_points waypoints smoothly, but still stops
            #   briefly at every call boundary (see _ptp9d_advance in
            #   inference_core.py).
            # service_stream = TRACK stage keeps a persistent PTP9D queue
            #   topped up (see _ptp9d_stream_topup); the robot never stops
            #   between calls, only if the queue actually runs dry. This is
            #   the only mode with true call-to-call continuity.
            # topic_publish = legacy continuous 9D command streaming
            #   directly onto cmd_topic at control_hz.
            DeclareLaunchArgument(
                "inference_mode",
                default_value="service_call",
                choices=["service_call", "service_stream", "topic_publish"],
            ),
            # Each PTP9D call now carries ptp9d_segment_points consecutive
            # lookahead waypoints (ptp9d_segment_stride raw samples apart),
            # blended into one continuous robot-side motion instead of
            # stopping fully at every point. Raise segment_points for
            # smoother/less "stop-start" motion at the cost of coarser
            # per-call contact/safety re-evaluation. Only used when
            # inference_mode=service_call.
            DeclareLaunchArgument("ptp9d_segment_points", default_value="15"),
            DeclareLaunchArgument("ptp9d_segment_stride", default_value="1"),
            # Only used when inference_mode=service_stream.
            DeclareLaunchArgument("ptp9d_stream_topup_points", default_value="20"),
            DeclareLaunchArgument("ptp9d_stream_min_lookahead_sec", default_value="1.0"),
            DeclareLaunchArgument("ptp9d_target_velocity_mm_s", default_value="10.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "ckpt_dir": ckpt_dir,
                    "act_root": act_root,
                    "policy_class": policy_class,
                    "ckpt_auto_subdir": ckpt_auto_subdir,
                    "metrics_log_enable": metrics_log_enable,
                    "metrics_log_dir": metrics_log_dir,
                    "metrics_run_tag": metrics_run_tag,
                    "pose_topic": pose_topic,
                    "force_topic": force_topic,
                    "image_topic": image_topic,
                    # Operational mode. Legacy ACT-era recovery heuristics are
                    # bypassed, but command smoothing and envelope safety stay on.
                    "visualization_only": "false",
                    "clean_flow_execution": "true",
                    "flow_diagnostic_only": "false",
                    "flow_step_service_enable": "false",
                    "auto_move_to_demo_start": "true",
                    "orientation_lock_enable": "false",
                    "contact_z_descent_block_enable": "true",
                    "force_xy_cmd_enable": "false",
                    "cmd_safety_enable": "true",
                    # Match the checkpoint's observation construction.
                    "use_stain_mask": use_stain_mask,
                    "track_use_ptp9d_service": track_use_ptp9d_service,
                    "ptp9d_use_stream": ptp9d_use_stream,
                    "ptp9d_segment_points": ptp9d_segment_points,
                    "ptp9d_segment_stride": ptp9d_segment_stride,
                    "ptp9d_stream_topup_points": LaunchConfiguration("ptp9d_stream_topup_points"),
                    "ptp9d_stream_min_lookahead_sec": LaunchConfiguration("ptp9d_stream_min_lookahead_sec"),
                    "ptp9d_target_velocity_mm_s": ptp9d_target_velocity_mm_s,
                    "auto_stain_mask": "true",
                    "stain_mask_mode": "tcp_roi",
                    "tcp_roi_reference_width": "424",
                    "tcp_roi_reference_height": "240",
                    "tcp_roi_center_x": "253",
                    "tcp_roi_center_y": "120",
                    # Must match the checkpoint's training-time tcp_roi_area_fraction
                    # (inference_core.py overrides the model's own value from
                    # dataset_stats.pkl, but stain_mask_publisher only sees this
                    # launch arg -- keep them in sync or the overlay box drifts
                    # from what the model actually attends to).
                    "tcp_roi_area_fraction": "0.25",
                    "camera_preprocess_mode": "stabilize",
                    "chunk_size": "128",
                    "use_force_history": "true",
                    "force_history_len": "30",
                    "flow_infer_steps": "10",
                    "flow_deterministic_noise": "true",
                    "flow_noise_seed": "0",
                    # Replay the learned FLOW trajectory directly. Let each
                    # absolute-referenced plan run close to its full
                    # chunk_size=128 horizon (~4.3s @ 30Hz) before replanning,
                    # instead of cutting it off after ~1s. local_anchor is off:
                    # anchoring plans to the live pose removed the self-
                    # correction against the model's absolute target, so a
                    # small per-step z bias accumulated into unbounded ascent
                    # (20260811 FLOW anchorfix runs, ESTOP both times).
                    "action_selection_mode": "trajectory_interp",
                    "trajectory_hz": "30.0",
                    "flow_local_anchor_enable": "false",
                    "flow_replan_interval_steps": "120",
                    # Only the requested diagnostic windows are shown.
                    "gradcam_enable": "false",
                    "visualize": "false",
                    "modality_importance_enable": "true",
                    "modality_importance_every_n_infer": modality_every_n,
                    "modality_importance_target": "action_norm",
                    "modality_importance_target_step": "0",
                    "modality_importance_target_horizon": "16",
                    "visualize_modality_importance": "true",
                    "flow_vector_overlay_enable": "true",
                    "flow_vector_overlay_horizons": "1,5,15,30,60,127",
                    "flow_vector_overlay_selected_horizon": vector_horizon,
                    "flow_vector_overlay_tcp_center_x": "253",
                    "flow_vector_overlay_tcp_center_y": "120",
                    "flow_vector_overlay_pixels_per_mm": "2.0",
                    "visualize_flow_vector": "true",
                }.items(),
            ),
        ]
    )
