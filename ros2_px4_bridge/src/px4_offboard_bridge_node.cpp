/**
 * AeroHarvest: PX4 Micro-XRCE-DDS Offboard Bridge Node (ROS 2 Humble / C++)
 * Author: Santhosh Kumar (Reg No: 23BAI1236) - Control & Autopilot Lead
 * Reference: PRD Section 2, 7 & 9 (Risk R2 Mitigation)
 * Translates velocity commands into PX4 Offboard Trajectory Setpoints
 * and handles offboard mode heartbeat with automatic failsafe fallback.
 */

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/bool.hpp"

using namespace std::chrono_literals;

class PX4OffboardBridgeNode : public rclcpp::Node {
public:
  PX4OffboardBridgeNode() : Node("px4_offboard_bridge_node") {
    // Subscribers
    cmd_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
      "/aeroharvest/cmd_vel", 10,
      std::bind(&PX4OffboardBridgeNode::cmdCallback, this, std::placeholders::_1));

    // Publishers to PX4 uORB topics via Micro-XRCE-DDS agent
    offboard_sp_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/fmu/in/trajectory_setpoint", 10);

    heartbeat_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/fmu/in/offboard_control_mode", 10);

    // 50 Hz Heartbeat timer (PX4 requires continuous stream > 2 Hz to stay in offboard mode)
    heartbeat_timer_ = this->create_wall_timer(
      20ms, std::bind(&PX4OffboardBridgeNode::publishHeartbeat, this));

    RCLCPP_INFO(this->get_logger(), "PX4 Offboard Bridge Node Initialized with DDS Keepalive.");
  }

private:
  void cmdCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
    last_cmd_ = *msg;
    last_cmd_time_ = this->now();
  }

  void publishHeartbeat() {
    auto hb = std_msgs::msg::Bool();
    hb.data = true;
    heartbeat_pub_->publish(hb);

    // Watchdog: If command drops for > 500ms, command zero velocity / altitude hold (Risk R2 mitigation)
    double dt_since_last_cmd = (this->now() - last_cmd_time_).seconds();
    auto sp = geometry_msgs::msg::TwistStamped();
    sp.header.stamp = this->now();
    sp.header.frame_id = "inertial_ned";

    if (dt_since_last_cmd < 0.5) {
      sp.twist = last_cmd_.twist;
    } else {
      // Safe hold hover
      sp.twist.linear.x = 0.0;
      sp.twist.linear.y = 0.0;
      sp.twist.linear.z = 0.0;
      sp.twist.angular.z = 0.0;
    }

    offboard_sp_pub_->publish(sp);
  }

  geometry_msgs::msg::TwistStamped last_cmd_;
  rclcpp::Time last_cmd_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr offboard_sp_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr heartbeat_pub_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PX4OffboardBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
