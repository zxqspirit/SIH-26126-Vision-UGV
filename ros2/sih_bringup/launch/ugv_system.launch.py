#!/usr/bin/env python3
"""Central UGV system launch file orchestrating the 7 minimum functional nodes."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1. Perception
        Node(
            package='sih_perception',
            executable='traversability_classifier_node',
            name='traversability_classifier_node',
            output='screen',
        ),

        # 2. Depth Geometry
        Node(
            package='sih_depth',
            executable='depth_geometry_node',
            name='depth_geometry_node',
            output='screen',
        ),

        # 3. Terrain Costmap Fusion
        Node(
            package='sih_fusion',
            executable='terrain_fusion_node',
            name='terrain_fusion_node',
            output='screen',
        ),

        # 4. Localization
        Node(
            package='sih_localization',
            executable='visual_odometry_node',
            name='visual_odometry_node',
            output='screen',
        ),

        # 5. Local Navigation
        Node(
            package='sih_navigation',
            executable='local_planner_node',
            name='local_planner_node',
            output='screen',
        ),

        # 6. Safety Supervisor
        Node(
            package='sih_safety',
            executable='safety_supervisor_node',
            name='safety_supervisor_node',
            output='screen',
        ),

        # 7. Visualization Bridge
        Node(
            package='sih_visualization',
            executable='telemetry_bridge_node',
            name='telemetry_bridge_node',
            output='screen',
        ),
    ])
