"""ROS 2 Launch file for beginner beacon publisher/subscriber exercise."""

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    log_notice = LogInfo(msg="Launching SIH 26126 Beginner ROS 2 Exercise (sih_starter)")

    # 1. Start the publisher node
    publisher_node = Node(
        package='sih_starter',
        executable='beacon_publisher',
        name='beacon_publisher',
        output='screen',
        parameters=[{
            'publish_period_sec': 1.0,
        }],
    )

    # 2. Start the subscriber node
    subscriber_node = Node(
        package='sih_starter',
        executable='beacon_subscriber',
        name='beacon_subscriber',
        output='screen',
    )

    return LaunchDescription([
        log_notice,
        publisher_node,
        subscriber_node,
    ])
