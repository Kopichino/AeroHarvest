/**
 * AeroHarvest: Gazebo Harmonic Canyon Wind Plugin Implementation
 * Author: Koppesh P (Reg No: 23BAI1113)
 * Reference: PRD Section 1.1, 2 & 7
 */

#include "CanyonWindPlugin.hh"
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/ExternalWorldWrenchCmd.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/plugin/Register.hh>
#include <cmath>

namespace aeroharvest {
namespace gazebo {

CanyonWindPlugin::CanyonWindPlugin() : rng_(42) {
  base_direction_.Normalize();
}

void CanyonWindPlugin::Configure(const gz::sim::Entity &_entity,
                                 const std::shared_ptr<const sdf::Element> &_sdf,
                                 gz::sim::EntityComponentManager &_ecm,
                                 gz::sim::EventManager &) {
  model_ = gz::sim::Model(_entity);
  link_entity_ = model_.LinkByName(_ecm, "base_link");

  if (_sdf->HasElement("base_wind_speed")) {
    base_wind_speed_ = _sdf->Get<double>("base_wind_speed");
  }
  if (_sdf->HasElement("gust_amplitude")) {
    gust_amplitude_ = _sdf->Get<double>("gust_amplitude");
  }
}

void CanyonWindPlugin::PreUpdate(const gz::sim::UpdateInfo &_info,
                                 gz::sim::EntityComponentManager &_ecm) {
  if (_info.paused) return;

  auto pose_comp = _ecm.Component<gz::sim::components::Pose>(link_entity_);
  if (!pose_comp) return;

  auto pos = pose_comp->Data().Pos();
  double x = pos.X();
  double y = pos.Y();
  double z = pos.Z();
  double t = std::chrono::duration<double>(_info.simTime).count();

  // 1. Canyon Channeling Draft (Venturi jet)
  double y_center = 80.0 * std::sin(0.004 * x);
  double dist_canyon = std::abs(y - y_center);
  double canyon_factor = std::exp(-std::pow(dist_canyon / 90.0, 2)) * std::exp(-std::pow(z / 120.0, 2));
  gz::math::Vector3d v_channel = canyon_factor * 4.5 * gz::math::Vector3d(0.85, 0.52, 0.1);

  // 2. Spatial micro-gust bursts
  double gust_mod = std::sin(2.0 * M_PI * gust_frequency_hz_ * t);
  gz::math::Vector3d v_gust = gust_amplitude_ * gust_mod * gz::math::Vector3d(
    std::cos(0.01 * x + t),
    std::sin(0.01 * y + t),
    0.4 * std::sin(2.0 * M_PI * gust_frequency_hz_ * t)
  );

  // 3. Base wind with elevation shear
  double z_norm = std::clamp(z / 150.0, 0.2, 2.5);
  gz::math::Vector3d v_wind = (base_wind_speed_ * std::pow(z_norm, 0.25)) * base_direction_ + v_channel + v_gust;

  // 4. Apply Aerodynamic drag wrench to hexacopter airframe: F = 0.5 * rho * Cd * A * ||v_rel|| * v_rel
  double rho = 1.225;
  double cd_area = 0.45 * 0.12;
  double v_norm = v_wind.Length();
  gz::math::Vector3d f_drag = 0.5 * rho * cd_area * v_norm * v_wind;

  gz::sim::components::ExternalWorldWrenchCmd wrench_cmd;
  wrench_cmd.Data().mutable_force()->set_x(f_drag.X());
  wrench_cmd.Data().mutable_force()->set_y(f_drag.Y());
  wrench_cmd.Data().mutable_force()->set_z(f_drag.Z());

  _ecm.SetComponentData<gz::sim::components::ExternalWorldWrenchCmd>(link_entity_, wrench_cmd);
}

} // namespace gazebo
} // namespace aeroharvest

GZ_ADD_PLUGIN(
  aeroharvest::gazebo::CanyonWindPlugin,
  gz::sim::System,
  aeroharvest::gazebo::CanyonWindPlugin::ISystemConfigure,
  aeroharvest::gazebo::CanyonWindPlugin::ISystemPreUpdate
)
