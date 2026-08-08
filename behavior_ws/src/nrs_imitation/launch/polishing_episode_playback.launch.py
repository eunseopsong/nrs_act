#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    episode_file = LaunchConfiguration("episode_file")
    dataset_hz = LaunchConfiguration("dataset_hz")
    playback_speed = LaunchConfiguration("playback_speed")
    control_hz = LaunchConfiguration("control_hz")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    use_action_force = LaunchConfiguration("use_action_force")
    force_xy_cmd_enable = LaunchConfiguration("force_xy_cmd_enable")
    max_command_fz_N = LaunchConfiguration("max_command_fz_N")
    measured_fz_abort_N = LaunchConfiguration("measured_fz_abort_N")
    allow_multiple_cmd_publishers = LaunchConfiguration(
        "allow_multiple_cmd_publishers"
    )
    align_max_distance_mm = LaunchConfiguration("align_max_distance_mm")
    align_max_xyz_speed_mm_s = LaunchConfiguration("align_max_xyz_speed_mm_s")
    align_max_rot_speed_rad_s = LaunchConfiguration("align_max_rot_speed_rad_s")
    observation_ratio_topic = LaunchConfiguration("observation_ratio_topic")
    visualize_observation_ratio = LaunchConfiguration("visualize_observation_ratio")

    default_episode = (
        "/home/eunseop/nrs_imitation/datasets/polishing/single_cam/20260801_2233/"
        "imitation_form_tcp_roi_square10pct_action_fxy_zero/episode_0.hdf5"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("episode_file", default_value=default_episode),
            DeclareLaunchArgument("dataset_hz", default_value="30.0"),
            DeclareLaunchArgument("playback_speed", default_value="1.0"),
            DeclareLaunchArgument("control_hz", default_value="125.0"),
            DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
            DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
            DeclareLaunchArgument("cmd_topic", default_value="/ur10skku/cmdMotion"),
            # Start with geometry-only playback. Set true explicitly only after
            # confirming the ep0 pose path; recorded ep0 Fz peaks at 31.5 N.
            DeclareLaunchArgument("use_action_force", default_value="false"),
            DeclareLaunchArgument("force_xy_cmd_enable", default_value="false"),
            DeclareLaunchArgument("max_command_fz_N", default_value="35.0"),
            DeclareLaunchArgument("measured_fz_abort_N", default_value="45.0"),
            DeclareLaunchArgument("allow_multiple_cmd_publishers", default_value="false"),
            DeclareLaunchArgument("align_max_distance_mm", default_value="150.0"),
            DeclareLaunchArgument("align_max_xyz_speed_mm_s", default_value="20.0"),
            DeclareLaunchArgument("align_max_rot_speed_rad_s", default_value="0.15"),
            DeclareLaunchArgument(
                "observation_ratio_topic",
                default_value="/polishing_episode_playback/observation_ratio",
            ),
            DeclareLaunchArgument("visualize_observation_ratio", default_value="true"),
            Node(
                package="nrs_imitation",
                executable="episode_playback",
                name="polishing_episode_playback",
                output="screen",
                parameters=[
                    {
                        "episode_file": episode_file,
                        "dataset_hz": ParameterValue(dataset_hz, value_type=float),
                        "playback_speed": ParameterValue(
                            playback_speed, value_type=float
                        ),
                        "control_hz": ParameterValue(control_hz, value_type=float),
                        "pose_topic": pose_topic,
                        "force_topic": force_topic,
                        "cmd_topic": cmd_topic,
                        "use_action_force": ParameterValue(
                            use_action_force, value_type=bool
                        ),
                        "force_xy_cmd_enable": ParameterValue(
                            force_xy_cmd_enable, value_type=bool
                        ),
                        "max_command_fz_N": ParameterValue(
                            max_command_fz_N, value_type=float
                        ),
                        "measured_fz_abort_N": ParameterValue(
                            measured_fz_abort_N, value_type=float
                        ),
                        "allow_multiple_cmd_publishers": ParameterValue(
                            allow_multiple_cmd_publishers, value_type=bool
                        ),
                        "align_max_distance_mm": ParameterValue(
                            align_max_distance_mm, value_type=float
                        ),
                        "align_max_xyz_speed_mm_s": ParameterValue(
                            align_max_xyz_speed_mm_s, value_type=float
                        ),
                        "align_max_rot_speed_rad_s": ParameterValue(
                            align_max_rot_speed_rad_s, value_type=float
                        ),
                        "observation_ratio_topic": observation_ratio_topic,
                    }
                ],
            ),
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                name="episode_playback_observation_ratio_viewer",
                output="screen",
                arguments=[observation_ratio_topic],
                condition=IfCondition(visualize_observation_ratio),
            ),
        ]
    )
