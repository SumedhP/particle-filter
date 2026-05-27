#pragma once

#include <concepts>

namespace pf::filter::concepts {

template <typename T>
concept initialization_prior = requires(const T p) {
  { p.robot_radius } -> std::convertible_to<float>;
};

}  // namespace pf::filter::concepts
