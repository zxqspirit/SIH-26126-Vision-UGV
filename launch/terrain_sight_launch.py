"""ROS 2 Jazzy Launch file for SIH 26126 Autonomous Navigation Stack.

Coordinates the software-only autonomous navigation nodes:
Perception -> Depth Geometry -> Fusion -> Localization -> Costmap & Planning -> Safety Gate -> cmd_vel.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    scenario_name = LaunchConfiguration('scenario', default='scenario_1_open_path')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true (default false for real sensor replay)'
    )

    declare_scenario = DeclareLaunchArgument(
        'scenario',
        default_value='scenario_1_open_path',
        description='Scenario name to replay from datasets/processed/'
    )

    log_banner = LogInfo(
        msg="Launching SIH 26126 Vision-Based Autonomous Navigation for Outdoor UGV (BEL)"
    )

    # Core Navigation Pipeline Node
    nav_pipeline_node = Node(
        package='sih_bringup',
        executable='navigation_pipeline_node',
        name='terrain_sight_navigation_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'scenario': scenario_name,
            'target_fps': 10.0,
        }],
        remappings=[
            ('/cmd_vel', '/ugv/cmd_vel'),
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_scenario,
        log_banner,
        nav_pipeline_node,
    ])
