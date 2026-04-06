#pragma once

#include <pf/config/target_config.h>

#include <cstddef>
#include <cstdint>

namespace pf::filter {

// particle_reduction_state<T> is the default accumulator used by configs that
// carry a full prediction object through the reduction tree.
//
// It satisfies the updated particle_reduction_operation concept via:
//   - state_type::zero()                      (unchanged)
//   - state_type::from_particle(const T&)     (new alias for from_one_particle)
//   - state.most_likely_particle()            (unchanged)
template <typename T>
struct particle_reduction_state {
  T most_likely_particle_;
  std::uint32_t count_;

  PF_TARGET_ATTRS [[nodiscard]] constexpr const T& most_likely_particle() const noexcept { return most_likely_particle_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const std::uint32_t& count() const noexcept { return count_; }

  PF_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> zero() noexcept {
    return particle_reduction_state<T>{T{}, std::uint32_t{}};
  }

  PF_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> from_one_particle(const T& particle) noexcept {
    return particle_reduction_state<T>{particle, std::uint32_t{1}};
  }

  // Alias required by the particle_reduction_operation concept.
  PF_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> from_particle(const T& particle) noexcept {
    return from_one_particle(particle);
  }
};

// Legacy transform functor — kept for any code that still uses it directly.
template <typename T>
struct particle_reduction_state_transform {
  PF_TARGET_ATTRS [[nodiscard]] inline particle_reduction_state<T> operator()(const T& particle) noexcept {
    return particle_reduction_state<T>::from_one_particle(particle);
  }
};

}  // namespace pf::filter
