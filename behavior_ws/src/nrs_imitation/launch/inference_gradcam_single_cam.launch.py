#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ckpt_dir = LaunchConfiguration("ckpt_dir")
    act_root = LaunchConfiguration("act_root")
    policy_class = LaunchConfiguration("policy_class")
    ckpt_auto_subdir = LaunchConfiguration("ckpt_auto_subdir")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    image_topic = LaunchConfiguration("image_topic")
    use_stain_mask = LaunchConfiguration("use_stain_mask")
    stain_mask_topic = LaunchConfiguration("stain_mask_topic")
    auto_stain_mask = LaunchConfiguration("auto_stain_mask")
    stain_mask_overlay_topic = LaunchConfiguration("stain_mask_overlay_topic")
    publish_stain_mask_overlay = LaunchConfiguration("publish_stain_mask_overlay")
    stain_mask_mode = LaunchConfiguration("stain_mask_mode")
    task_roi_center_x = LaunchConfiguration("task_roi_center_x")
    task_roi_y_end = LaunchConfiguration("task_roi_y_end")
    task_roi_half_width = LaunchConfiguration("task_roi_half_width")
    tcp_roi_reference_width = LaunchConfiguration("tcp_roi_reference_width")
    tcp_roi_reference_height = LaunchConfiguration("tcp_roi_reference_height")
    tcp_roi_center_x = LaunchConfiguration("tcp_roi_center_x")
    tcp_roi_center_y = LaunchConfiguration("tcp_roi_center_y")
    tcp_roi_area_fraction = LaunchConfiguration("tcp_roi_area_fraction")
    stain_dark_thresh = LaunchConfiguration("stain_dark_thresh")
    reflection_v_thresh = LaunchConfiguration("reflection_v_thresh")
    reflection_s_thresh = LaunchConfiguration("reflection_s_thresh")
    stain_min_area = LaunchConfiguration("stain_min_area")
    stain_morph_kernel = LaunchConfiguration("stain_morph_kernel")
    cmd_topic = LaunchConfiguration("cmd_topic")
    camera_preprocess_mode = LaunchConfiguration("camera_preprocess_mode")
    chunk_size = LaunchConfiguration("chunk_size")
    use_force_history = LaunchConfiguration("use_force_history")
    force_history_len = LaunchConfiguration("force_history_len")
    flow_infer_steps = LaunchConfiguration("flow_infer_steps")

    gradcam_enable = LaunchConfiguration("gradcam_enable")
    gradcam_publish = LaunchConfiguration("gradcam_publish")
    gradcam_every_n_infer = LaunchConfiguration("gradcam_every_n_infer")
    gradcam_target = LaunchConfiguration("gradcam_target")
    gradcam_target_step = LaunchConfiguration("gradcam_target_step")
    gradcam_target_horizon = LaunchConfiguration("gradcam_target_horizon")
    gradcam_layer_name = LaunchConfiguration("gradcam_layer_name")
    gradcam_overlay_topic = LaunchConfiguration("gradcam_overlay_topic")
    gradcam_save = LaunchConfiguration("gradcam_save")
    gradcam_save_dir = LaunchConfiguration("gradcam_save_dir")
    visualize = LaunchConfiguration("visualize")

    return LaunchDescription([
        DeclareLaunchArgument("ckpt_dir", default_value=""),
        DeclareLaunchArgument("act_root", default_value="~/nrs_imitation"),
        DeclareLaunchArgument("policy_class", default_value="FLOW"),
        DeclareLaunchArgument("ckpt_auto_subdir", default_value="polishing/single_cam"),
        DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
        DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
        DeclareLaunchArgument("image_topic", default_value="/realsense/vr/color/image_raw"),
        DeclareLaunchArgument("use_stain_mask", default_value="false"),
        DeclareLaunchArgument("stain_mask_topic", default_value="/inference_single_cam/stain_mask"),
        DeclareLaunchArgument("auto_stain_mask", default_value="false"),
        DeclareLaunchArgument("stain_mask_overlay_topic", default_value="/inference_single_cam/stain_mask_overlay"),
        DeclareLaunchArgument("publish_stain_mask_overlay", default_value="true"),
        DeclareLaunchArgument("stain_mask_mode", default_value="rgb_threshold"),
        DeclareLaunchArgument("task_roi_center_x", default_value="253"),
        DeclareLaunchArgument("task_roi_y_end", default_value="110"),
        DeclareLaunchArgument("task_roi_half_width", default_value="12"),
        DeclareLaunchArgument("tcp_roi_reference_width", default_value="424"),
        DeclareLaunchArgument("tcp_roi_reference_height", default_value="240"),
        DeclareLaunchArgument("tcp_roi_center_x", default_value="253"),
        DeclareLaunchArgument("tcp_roi_center_y", default_value="120"),
        DeclareLaunchArgument("tcp_roi_area_fraction", default_value="0.10"),
        DeclareLaunchArgument("stain_dark_thresh", default_value="80"),
        DeclareLaunchArgument("reflection_v_thresh", default_value="235"),
        DeclareLaunchArgument("reflection_s_thresh", default_value="60"),
        DeclareLaunchArgument("stain_min_area", default_value="20"),
        DeclareLaunchArgument("stain_morph_kernel", default_value="3"),
        DeclareLaunchArgument("cmd_topic", default_value="/ur10skku/cmdMotion"),
        DeclareLaunchArgument("camera_preprocess_mode", default_value="stabilize"),
        DeclareLaunchArgument("chunk_size", default_value="200"),
        DeclareLaunchArgument("use_force_history", default_value="true"),
        DeclareLaunchArgument("force_history_len", default_value="10"),
        DeclareLaunchArgument("flow_infer_steps", default_value="10"),

        DeclareLaunchArgument("gradcam_enable", default_value="true"),
        DeclareLaunchArgument("gradcam_publish", default_value="true"),
        DeclareLaunchArgument("gradcam_every_n_infer", default_value="1"),
        DeclareLaunchArgument("gradcam_target", default_value="z"),
        DeclareLaunchArgument("gradcam_target_step", default_value="0"),
        DeclareLaunchArgument("gradcam_target_horizon", default_value="1"),
        DeclareLaunchArgument("gradcam_layer_name", default_value=""),
        DeclareLaunchArgument("gradcam_overlay_topic", default_value="/inference_single_cam/gradcam_overlay"),
        DeclareLaunchArgument("gradcam_save", default_value="false"),
        DeclareLaunchArgument("gradcam_save_dir", default_value="~/nrs_imitation/gradcam"),
        DeclareLaunchArgument("visualize", default_value="true"),

        Node(
            package="nrs_imitation",
            executable="stain_mask_publisher",
            name="stain_mask_publisher",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
                "mask_topic": stain_mask_topic,
                "overlay_topic": stain_mask_overlay_topic,
                "publish_overlay": ParameterValue(publish_stain_mask_overlay, value_type=bool),
                "mask_mode": stain_mask_mode,
                "task_roi_center_x": ParameterValue(task_roi_center_x, value_type=int),
                "task_roi_y_end": ParameterValue(task_roi_y_end, value_type=int),
                "task_roi_half_width": ParameterValue(task_roi_half_width, value_type=int),
                "tcp_roi_reference_width": ParameterValue(tcp_roi_reference_width, value_type=int),
                "tcp_roi_reference_height": ParameterValue(tcp_roi_reference_height, value_type=int),
                "tcp_roi_center_x": ParameterValue(tcp_roi_center_x, value_type=int),
                "tcp_roi_center_y": ParameterValue(tcp_roi_center_y, value_type=int),
                "tcp_roi_area_fraction": ParameterValue(tcp_roi_area_fraction, value_type=float),
                "stain_dark_thresh": ParameterValue(stain_dark_thresh, value_type=int),
                "reflection_v_thresh": ParameterValue(reflection_v_thresh, value_type=int),
                "reflection_s_thresh": ParameterValue(reflection_s_thresh, value_type=int),
                "stain_min_area": ParameterValue(stain_min_area, value_type=int),
                "stain_morph_kernel": ParameterValue(stain_morph_kernel, value_type=int),
            }],
            condition=IfCondition(auto_stain_mask),
        ),

        Node(
            package="nrs_imitation",
            executable="inference_single_cam",
            name="inference_single_cam",
            output="screen",
            parameters=[{
                "ckpt_dir": ckpt_dir,
                "act_root": act_root,
                "policy_class": policy_class,
                "ckpt_auto_subdir": ckpt_auto_subdir,
                "pose_topic": pose_topic,
                "force_topic": force_topic,
                "image_topic": image_topic,
                # ROS/YAML interprets the unquoted token "off" as Boolean False.
                # Force a string so camera_preprocess_mode:=off reaches the node as intended.
                "camera_preprocess_mode": ParameterValue(camera_preprocess_mode, value_type=str),
                "chunk_size": ParameterValue(chunk_size, value_type=int),
                "use_force_history": ParameterValue(use_force_history, value_type=bool),
                "force_history_len": ParameterValue(force_history_len, value_type=int),
                "flow_infer_steps": ParameterValue(flow_infer_steps, value_type=int),
                "use_stain_mask": ParameterValue(use_stain_mask, value_type=bool),
                "stain_mask_topic": stain_mask_topic,
                "cmd_topic": cmd_topic,
                "gradcam_enable": ParameterValue(gradcam_enable, value_type=bool),
                "gradcam_publish": ParameterValue(gradcam_publish, value_type=bool),
                "gradcam_every_n_infer": ParameterValue(gradcam_every_n_infer, value_type=int),
                "gradcam_target": gradcam_target,
                "gradcam_target_step": ParameterValue(gradcam_target_step, value_type=int),
                "gradcam_target_horizon": ParameterValue(gradcam_target_horizon, value_type=int),
                "gradcam_layer_name": gradcam_layer_name,
                "gradcam_overlay_topic": gradcam_overlay_topic,
                "gradcam_save": ParameterValue(gradcam_save, value_type=bool),
                "gradcam_save_dir": gradcam_save_dir,
            }],
        ),

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="gradcam_viewer",
            output="screen",
            arguments=[gradcam_overlay_topic],
            condition=IfCondition(visualize),
        ),
    ])
