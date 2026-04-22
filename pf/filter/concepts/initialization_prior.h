#pragma once

#include <concepts>
#include <type_traits>

namespace pf::filter::concepts {

template <typename...>
using void_t = void;

template <typename T>
concept has_initialization_prior_type = requires {
  typename T::initialization_prior_type;
};

template <typename T>
concept initialization_prior_compatible =
    has_initialization_prior_type<T> &&
    requires(
        const T c,
        typename T::sampler_type s,
        const typename T::observation_type o,
        const typename T::initialization_prior_type p) {
      { c.sample_from(s, o, p) } -> std::same_as<typename T::prediction_type>;
    };

template <typename T, typename = void>
struct initialization_prior_type {
  using type = void;
};

template <typename T>
struct initialization_prior_type<T, void_t<typename T::initialization_prior_type>> {
  using type = typename T::initialization_prior_type;
};

template <typename T>
using initialization_prior_type_t = typename initialization_prior_type<T>::type;

}  // namespace pf::filter::concepts