/**
 * AeroHarvest: MPPI Local Trajectory Planner Node (ROS 2 Humble / C++)
 * Author: Santhosh Kumar (Reg No: 23BAI1236) - Control & Autopilot Lead
 * Reference: PRD Section 1.5, 3.3, 3.4 & Layer 5
 * Receives target waypoints, wind estimates, and executes 20 Hz MPPI trajectory rollouts.
 */

#include <chrono>
#include <vector>
#include <random>
#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"

using namespace std::chrono_literals;

class MPPIControllerNode : public rclcpp::Node {
public:
  MPPIControllerNode() : Node("mppi_controller_node"), rng_(42) {
    this->declare_parameter<int>("num_rollouts_k", 1024);
    this->declare_parameter<int>("horizon_steps_n", 30);
    this->declare_parameter<double>("dt_step", 0.2);
    this->declare_parameter<double>("temperature_lambda", 0.5);

    k_rollouts_ = this->get_parameter("num_rollouts_k").as_int();
    n_steps_ = this->get_parameter("horizon_steps_n").as_int();
    dt_ = this->get_parameter("dt_step").as_double();
    lambda_ = this->get_parameter("temperature_lambda").as_double();

    // Subscribers
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/px4/vehicle_odometry", 10,
      std::bind(&MPPIControllerNode::odomCallback, this, std::placeholders::_1));

    target_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/aeroharvest/target_setpoint", 10,
      std::bind(&MPPIControllerNode::targetCallback, this, std::placeholders::_1));

    wind_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/aeroharvest/wind_estimate", 10,
      std::bind(&MPPIControllerNode::windCallback, this, std::placeholders::_1));

    // Publisher to PX4 Offboard velocity setpoint
    cmd_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/aeroharvest/cmd_vel", 10);

    // 20 Hz loop (50 ms)
    control_timer_ = this->create_wall_timer(
      50ms, std::bind(&MPPIControllerNode::controlLoop, this));

    RCLCPP_INFO(this->get_logger(), "AeroHarvest MPPI Trajectory Controller Node Initialized (K=%d, N=%d).", k_rollouts_, n_steps_);
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    curr_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
    curr_vel_ << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z;
  }

  void targetCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    target_pos_ << msg->point.x, msg->point.y, msg->point.z;
  }

  void windCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
    wind_est_ << msg->vector.x, msg->vector.y, msg->vector.z;
  }

  void controlLoop() {
    // MPPI trajectory optimization over K candidate rollouts
    std::normal_distribution<double> dist_xy(0.0, 1.2);
    std::normal_distribution<double> dist_z(0.0, 0.6);

    std::vector<double> costs(k_rollouts_, 0.0);
    std::vector<Eigen::Vector3d> first_commands(k_rollouts_);

    double min_cost = 1e9;

    for (int k = 0; k < k_rollouts_; ++k) {
      Eigen::Vector3d p = curr_pos_;
      Eigen::Vector3d v = curr_vel_;
      double cost = 0.0;

      Eigen::Vector3d v_cmd_first = Eigen::Vector3d::Zero();

      for (int t = 0; t < n_steps_; ++t) {
        Eigen::Vector3d v_cmd = (target_pos_ - p).normalized() * 8.0;
        v_cmd.x() += dist_xy(rng_);
        v_cmd.y() += dist_xy(rng_);
        v_cmd.z() += dist_z(rng_);

        // Clamp
        v_cmd.x() = std::clamp(v_cmd.x(), -12.0, 12.0);
        v_cmd.y() = std::clamp(v_cmd.y(), -12.0, 12.0);
        v_cmd.z() = std::clamp(v_cmd.z(), -3.0, 4.0);

        if (t == 0) first_commands[k] = v_cmd;

        // Kinematics integration
        v += 2.5 * (v_cmd - v) * dt_;
        p += v * dt_;

        // Cost terms: relative wind power + goal distance
        Eigen::Vector3d v_rel = v - wind_est_;
        double p_elec = 380.0 + 0.5 * 1.225 * 0.45 * 0.12 * std::pow(v_rel.norm(), 3);
        cost += 0.04 * (p_elec / 100.0) * dt_;
      }

      cost += 15.0 * (p - target_pos_).norm();
      costs[k] = cost;
      if (cost < min_cost) min_cost = cost;
    }

    // Softmax weights
    double weight_sum = 0.0;
    std::vector<double> weights(k_rollouts_);
    for (int k = 0; k < k_rollouts_; ++k) {
      weights[k] = std::exp(-(costs[k] - min_cost) / lambda_);
      weight_sum += weights[k];
    }

    Eigen::Vector3d opt_cmd = Eigen::Vector3d::Zero();
    for (int k = 0; k < k_rollouts_; ++k) {
      opt_cmd += (weights[k] / (weight_sum + 1e-8)) * first_commands[k];
    }

    // Publish velocity command
    auto twist_msg = geometry_msgs::msg::TwistStamped();
    twist_msg.header.stamp = this->now();
    twist_msg.header.frame_id = "inertial_ned";
    twist_msg.twist.linear.x = opt_cmd.x();
    twist_msg.twist.linear.y = opt_cmd.y();
    twist_msg.twist.linear.z = opt_cmd.z();
    twist_msg.twist.angular.z = 0.0;

    cmd_pub_->publish(twist_msg);
  }

  int k_rollouts_{1024};
  int n_steps_{30};
  double dt_{0.2};
  double lambda_{0.5};

  std::mt19937 rng_;
  Eigen::Vector3d curr_pos_{0.0, 0.0, 30.0};
  Eigen::Vector3d curr_vel_{0.0, 0.0, 0.0};
  Eigen::Vector3d target_pos_{0.0, 0.0, 30.0};
  Eigen::Vector3d wind_est_{0.0, 0.0, 0.0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr wind_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPPIControllerNode>());
  rclcpp::shutdown();
  return 0;
}
