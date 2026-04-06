#pragma once

#include <pf/filter/concepts/observation.h>
#include <pf/filter/concepts/prediction.h>
#include <pf/filter/particle_reduction_state.h>

#include <concepts>

namespace pf::filter::concepts {

template <typename T, typename P>
concept particle_reduction_operation =
    requires(const T op,
             const typename T::state_type s,
             const P p) {
      requires prediction<P>;

      // The operator must declare its accumulator type.
      typename T::state_type;

      // Identity element for the reduction tree.
      { T::state_type::zero() } -> std::same_as<typename T::state_type>;

      // Per-particle seed: wraps a single prediction into the accumulator.
      { T::state_type::from_particle(p) } -> std::same_as<typename T::state_type>;

      // Final extraction: produces a prediction from the accumulated state.
      // Called once after the full reduction is complete.
      { s.most_likely_particle() } -> std::convertible_to<P>;

      // Binary merge operator.
      { op(s, s) } -> std::same_as<typename T::state_type>;
    };

}  // namespace pf::filter::concepts
