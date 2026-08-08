#!/usr/bin/env python3
"""Read-only FLOW diagnostics: modality ratio + local-camera vector overlay."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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
            DeclareLaunchArgument(
                "act_root", default_value="/home/eunseop/nrs_imitation"
            ),
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
                    # Hard read-only boundary: no cmdMotion publisher and no
                    # control timer are created inside inference_core.
                    "visualization_only": "true",
                    "auto_move_to_demo_start": "false",
                    "orientation_lock_enable": "false",
                    "flow_diagnostic_only": "true",
                    "flow_step_service_enable": "false",
                    # Observation path must match the checkpoint training setup.
                    "use_stain_mask": use_stain_mask,
                    "auto_stain_mask": "true",
                    "stain_mask_mode": "tcp_roi",
                    "tcp_roi_reference_width": "424",
                    "tcp_roi_reference_height": "240",
                    "tcp_roi_center_x": "253",
                    "tcp_roi_center_y": "120",
                    "tcp_roi_area_fraction": "0.10",
                    "camera_preprocess_mode": "stabilize",
                    "chunk_size": "128",
                    "use_force_history": "true",
                    "force_history_len": "30",
                    "flow_infer_steps": "10",
                    "flow_deterministic_noise": "true",
                    "flow_noise_seed": "0",
                    "action_selection_mode": "trajectory_interp",
                    "trajectory_hz": "30.0",
                    "flow_local_anchor_enable": "false",
                    "flow_replan_interval_steps": "0",
                    # Keep only the two requested diagnostics.
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
