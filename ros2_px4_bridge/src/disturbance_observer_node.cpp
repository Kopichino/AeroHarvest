/**
 * AeroHarvest: 50 Hz Telemetry-Based Disturbance Observer Node (ROS 2 Humble / C++)
 * Author: Shaik Abdulkhader Masthan (Reg No: 23BAI1123) - Perception & ML Lead
 * Mathematical Reference: PRD Section 3.1
 * Inverts multirotor body drag equation:
 * F_ext = m * a_meas - R_b^I * [0, 0, T]^T + m * g * e3
 * v_rel = -sqrt(2 * ||F_ext|| / (rho * Cd * A)) * (F_ext / ||F_ext||)
 * v_wind = v_ground - v_rel
 */

#include <chrono>
#include <cmath>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;

class DisturbanceObserverNode : public rclcpp::Node {
public:
  DisturbanceObserverNode() : Node("disturbance_observer_node") {
    // Parameters
    this->declare_parameter<double>("mass_kg", 2.4);
    this->declare_parameter<double>("air_density_kg_m3", 1.225);
    this->declare_parameter<double>("drag_coeff_cd", 0.45);
    this->declare_parameter<double>("projected_area_m2", 0.12);
    this->declare_parameter<double>("filter_alpha", 0.25);

    mass_ = this->get_parameter("mass_kg").as_double();
    rho_ = this->get_parameter("air_density_kg_m3").as_double();
    cd_ = this->get_parameter("drag_coeff_cd").as_double();
    area_ = this->get_parameter("projected_area_m2").as_double();
    alpha_ = this->get_parameter("filter_alpha").as_double();

    drag_factor_ = rho_ * cd_ * area_;
    v_wind_filtered_.setZero();

    // Subscriptions
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/px4/sensor_combined", 10,
      std::bind(&DisturbanceObserverNode::imuCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/px4/vehicle_odometry", 10,
      std::bind(&DisturbanceObserverNode::odomCallback, this, std::placeholders::_1));

    thrust_sub_ = this->create_subscription<std_msgs::msg::Float32>(
      "/px4/vehicle_thrust_telemetry", 10,
      std::bind(&DisturbanceObserverNode::thrustCallback, this, std::placeholders::_1));

    // Publishers
    wind_pub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
      "/aeroharvest/wind_estimate", 10);
    f_ext_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/aeroharvest/external_force", 10);

    // 50 Hz timer execution loop (20 ms period)
    timer_ = this->create_wall_timer(
      20ms, std::bind(&DisturbanceObserverNode::computeObserverStep, this));

    RCLCPP_INFO(this->get_logger(), "AeroHarvest 50 Hz C++ Telemetry Disturbance Observer Initialized.");
  }

private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    latest_accel_ << msg->linear_acceleration.x,
                     msg->linear_acceleration.y,
                     msg->linear_acceleration.z;

    Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
    latest_R_b_I_ = q.toRotationMatrix();
    has_imu_ = true;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    latest_v_ground_ << msg->twist.twist.linear.x,
                        msg->twist.twist.linear.y,
                        msg->twist.twist.linear.z;
    has_odom_ = true;
  }

  void thrustCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    latest_thrust_ = msg->data;
  }

  void computeObserverStep() {
    if (!has_imu_ || !has_odom_) {
      return;
    }

    // Body thrust vector along z-axis: R_b^I * [0, 0, T]^T
    Eigen::Vector3d thrust_body(0.0, 0.0, latest_thrust_);
    Eigen::Vector3d thrust_inertial = latest_R_b_I_ * thrust_body;

    // PRD Eq 104: F_ext = m * a_meas - thrust_inertial + m * g * e3
    // Note: For specific force measurement from accelerometer, gravity is already sensed
    Eigen::Vector3d f_ext = mass_ * latest_accel_ - thrust_inertial;

    double f_norm = f_ext.norm();
    Eigen::Vector3d v_rel = Eigen::Vector3d::Zero();

    if (f_norm > 1e-4) {
      double speed_rel = std::sqrt((2.0 * f_norm) / (drag_factor_ + 1e-6));
      speed_rel = std::min(speed_rel, 30.0);
      v_rel = -speed_rel * (f_ext / f_norm);
    }

    // PRD Eq 120: v_wind = v_ground - v_rel
    Eigen::Vector3d v_wind_raw = latest_v_ground_ - v_rel;

    // Exponential smoothing filter
    if (!initialized_) {
      v_wind_filtered_ = v_wind_raw;
      initialized_ = true;
    } else {
      v_wind_filtered_ = alpha_ * v_wind_raw + (1.0 - alpha_) * v_wind_filtered_;
    }

    // Publish estimated wind vector
    auto wind_msg = geometry_msgs::msg::Vector3Stamped();
    wind_msg.header.stamp = this->now();
    wind_msg.header.frame_id = "inertial_ned";
    wind_msg.vector.x = v_wind_filtered_.x();
    wind_msg.vector.y = v_wind_filtered_.y();
    wind_msg.vector.z = v_wind_filtered_.z();
    wind_pub_->publish(wind_msg);

    // Publish external aerodynamic force
    auto wrench_msg = geometry_msgs::msg::WrenchStamped();
    wrench_msg.header = wind_msg.header;
    wrench_msg.wrench.force.x = f_ext.x();
    wrench_msg.wrench.force.y = f_ext.y();
    wrench_msg.wrench.force.z = f_ext.z();
    f_ext_pub_->publish(wrench_msg);
  }

  double mass_, rho_, cd_, area_, alpha_, drag_factor_;
  double latest_thrust_{2.4 * 9.80665};
  bool has_imu_{false};
  bool has_odom_{false};
  bool initialized_{false};

  Eigen::Vector3d latest_accel_{0.0, 0.0, 9.80665};
  Eigen::Matrix3d latest_R_b_I_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d latest_v_ground_{0.0, 0.0, 0.0};
  Eigen::Vector3d v_wind_filtered_{0.0, 0.0, 0.0};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr thrust_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr wind_pub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr f_ext_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DisturbanceObserverNode>());
  rclcpp::shutdown();
  return 0;
}
