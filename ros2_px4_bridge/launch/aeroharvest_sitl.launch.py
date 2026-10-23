"""
AeroHarvest Full SITL ROS 2 Launch File.
Launches the 50 Hz Disturbance Observer, MPPI Controller, and PX4 Offboard Bridge.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1. 50 Hz Disturbance Observer Node
        Node(
            package='aeroharvest_bridge',
            executable='disturbance_observer_node',
            name='disturbance_observer_node',
            output='screen',
            parameters=[{
                'mass_kg': 2.4,
                'air_density_kg_m3': 1.225,
                'drag_coeff_cd': 0.45,
                'projected_area_m2': 0.12,
                'filter_alpha': 0.25,
            }],
        ),

        # 2. MPPI Trajectory Controller Node (20 Hz)
        Node(
            package='aeroharvest_bridge',
            executable='mppi_controller_node',
            name='mppi_controller_node',
            output='screen',
            parameters=[{
                'num_rollouts_k': 1024,
                'horizon_steps_n': 30,
                'dt_step': 0.2,
                'temperature_lambda': 0.5,
            }],
        ),

        # 3. PX4 Offboard Bridge Node
        Node(
            package='aeroharvest_bridge',
            executable='px4_offboard_bridge_node',
            name='px4_offboard_bridge_node',
            output='screen',
        ),
    ])
