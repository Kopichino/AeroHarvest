/**
 * AeroHarvest: Gazebo Harmonic 3D Mountain Canyon Wind Field Plugin (C++)
 * Author: Koppesh P (Reg No: 23BAI1113) - Simulation Physics Lead
 * Reference: PRD Section 1.1, 2 (Layer 1), 7
 * Generates spatially varying wind vectors, canyon draft Venturi jets,
 * and Gaussian micro-gusts applied directly to Gazebo multirotor body links.
 */

#ifndef CANYON_WIND_PLUGIN_HH_
#define CANYON_WIND_PLUGIN_HH_

#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/math/Vector3.hh>
#include <memory>
#include <random>

namespace aeroharvest {
namespace gazebo {

class CanyonWindPlugin : public gz::sim::System,
                         public gz::sim::ISystemConfigure,
                         public gz::sim::ISystemPreUpdate {
public:
  CanyonWindPlugin();
  ~CanyonWindPlugin() override = default;

  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &_eventMgr) override;

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override;

private:
  gz::sim::Model model_;
  gz::sim::Entity link_entity_;

  double base_wind_speed_{6.0};
  double gust_amplitude_{3.0};
  double gust_frequency_hz_{0.8};
  gz::math::Vector3d base_direction_{1.0, 1.0, 0.0};

  std::mt19937 rng_;
};

} // namespace gazebo
} // namespace aeroharvest

#endif // CANYON_WIND_PLUGIN_HH_
