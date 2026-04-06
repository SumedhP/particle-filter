#pragma once

#include <pf/filter/concepts/observation.h>
#include <pf/filter/concepts/prediction.h>
#include <pf/filter/particle_reduction_state.h>

#include <concepts>

namespace pf::filter::concepts {

// ---------------------------------------------------------------------------
// particle_reduction_operation
//
// Previously this concept required the reduction operator to work over
// particle_reduction_state<P> specifically.  That forced every reduction
// to carry a full prediction object throughout the tree reduction, making
// a scalar SOA reduction impossible without first reconstructing a prediction.
//
// The updated concept requires only that the operator:
//   1. Declares a nested `state_type`.
//   2. Exposes a static `state_type::zero()` — the identity element.
//   3. Exposes a static `state_type::from_particle(const P&)` — constructs a
//      single-particle state from a prediction value.  This is called once per
//      particle in the transform step before the reduction tree.
//   4. The result of `state.most_likely_particle()` is convertible to P.
//      This is called exactly once at the end to extract the final answer.
//   5. `op(state, state) -> state_type` — the binary merge.
//
// Configurations that already use particle_reduction_state<P> satisfy this
// automatically because particle_reduction_state provides zero(),
// from_one_particle() (aliased as from_particle() below via the existing
// particle_reduction_state_transform shim), and most_likely_particle().
//
// New configurations (e.g. scalar SOA reductions) provide their own
// state_type with the same four-point interface.
// ---------------------------------------------------------------------------
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
