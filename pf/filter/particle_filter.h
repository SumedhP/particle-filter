#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/concepts/initialization_prior.h>
#include <pf/filter/concepts/particle_filter_configuration.h>
#include <pf/filter/particle_reduction_state.h>
#include <pf/filter/systematic_resampler.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/disjoint_tls_pool.h>
#include <thrust/mr/new.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cuda/std/tuple>
#include <utility>

namespace pf::filter {

namespace helper {

using disjoint_pool_resource_type =
    thrust::mr::disjoint_unsynchronized_pool_resource<thrust::device_memory_resource, thrust::mr::new_delete_resource>;

using caching_allocator_type = thrust::mr::allocator<char, helper::disjoint_pool_resource_type>;

inline caching_allocator_type thread_local_caching_allocator() {
  return {&thrust::mr::tls_disjoint_pool(
      thrust::mr::get_global_resource<thrust::device_memory_resource>(),
      thrust::mr::get_global_resource<thrust::mr::new_delete_resource>())};
}

}  // namespace helper

template <typename ParticleFilterConfiguration>
  requires concepts::particle_filter_configuration<ParticleFilterConfiguration>
class particle_filter {
 private:
  using particle_configuration_type = ParticleFilterConfiguration;

  using observation_type    = typename ParticleFilterConfiguration::observation_type;
  using prediction_type     = typename ParticleFilterConfiguration::prediction_type;
  using sampler_type        = typename ParticleFilterConfiguration::sampler_type;
  using soa_storage_type    = typename prediction_type::soa_storage;

  using reduction_op_type    = decltype(std::declval<ParticleFilterConfiguration>().most_likely_particle_reduction());
  using reduction_state_type = typename reduction_op_type::state_type;

  using initialization_prior_type = concepts::initialization_prior_type_t<ParticleFilterConfiguration>;

  static constexpr bool supports_initialization_prior_v =
      concepts::initialization_prior_compatible<ParticleFilterConfiguration>;

  using initialization_prior_param_type =
      std::conditional_t<supports_initialization_prior_v, initialization_prior_type, observation_type>;

  helper::caching_allocator_type caching_allocator_;
  ParticleFilterConfiguration config_;

  prediction_type most_likely_particle_state_;
  systematic_resampler<prediction_type, std::uint32_t> resampler_;

  target_config::vector<sampler_type> sampler_states_;
  target_config::vector<float> log_particle_weights_;
  soa_storage_type particle_states_;

 public:
  [[nodiscard]] prediction_type extrapolate_state(const float& time_offset_seconds) const noexcept {
    return most_likely_particle_state_.extrapolate_state(time_offset_seconds);
  }

  void update_state_sans_observation(const float& time_offset_seconds) noexcept {
    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), particle_states_.zip_begin()),
        thrust::make_zip_iterator(sampler_states_.end(),   particle_states_.zip_end()),
        [config = config_, time_offset_seconds]
        PF_TARGET_ONLY_ATTRS(auto tuple) {
          sampler_type& sampler_state = cuda::std::get<0>(tuple);
          auto& field_tuple           = cuda::std::get<1>(tuple);

          prediction_type particle_state = prediction_type::from_soa_tuple(field_tuple);
          config.apply_process(time_offset_seconds, sampler_state, particle_state);
          prediction_type::to_soa_tuple(field_tuple, particle_state);
        });

    most_likely_particle_state_ = reduce_most_likely_();
  }

  void update_state_with_observation(const float& time_offset_seconds, const observation_type& observation_state) noexcept {
    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.zip_begin()),
        thrust::make_zip_iterator(sampler_states_.end(),   log_particle_weights_.end(),   particle_states_.zip_end()),
        [config = config_, time_offset_seconds, observation_state]
        PF_TARGET_ONLY_ATTRS(auto tuple) {
          sampler_type& sampler_state   = cuda::std::get<0>(tuple);
          float&        particle_weight = cuda::std::get<1>(tuple);
          auto&         field_tuple     = cuda::std::get<2>(tuple);

          prediction_type particle_state = prediction_type::from_soa_tuple(field_tuple);
          config.apply_process(time_offset_seconds, sampler_state, particle_state);
          particle_weight = config.conditional_log_likelihood(sampler_state, observation_state, particle_state);
          prediction_type::to_soa_tuple(field_tuple, particle_state);
        });

    resampler_.resample(target_config::policy(caching_allocator_), log_particle_weights_, particle_states_);
    most_likely_particle_state_ = reduce_most_likely_();
  }

  // Reinitialize around a new observation without reallocating or reseeding
  // sampler states; this reuses existing buffers and RNG progression.
  void reinitialize(const observation_type& initial_observation) noexcept {
    initialize_particle_states_(initial_observation);
  }

  void reinitialize(const observation_type& initial_observation, const initialization_prior_param_type& initialization_prior) noexcept
    requires supports_initialization_prior_v {
    initialize_particle_states_(initial_observation, initialization_prior);
  }

 public:
  [[nodiscard]] prediction_type reduce_most_likely_() noexcept {
    const auto reduction_op = config_.most_likely_particle_reduction();

    return thrust::transform_reduce(
               target_config::policy(caching_allocator_),
               particle_states_.zip_cbegin(),
               particle_states_.zip_cend(),
               [] PF_TARGET_ONLY_ATTRS(const auto& field_tuple) -> reduction_state_type {
                 return reduction_state_type::from_particle(
                     prediction_type::from_soa_tuple(field_tuple));
               },
               reduction_state_type::zero(),
               reduction_op)
        .most_likely_particle();
  }

  void initialize_sampler_states_(const std::size_t& number_of_particles) noexcept {
    thrust::counting_iterator<std::size_t> index_sequence_begin(std::size_t{});

    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin()),
        thrust::make_zip_iterator(index_sequence_begin + number_of_particles, sampler_states_.end()),
        [number_of_particles] PF_TARGET_ATTRS(cuda::std::tuple<std::size_t, sampler_type&> tuple) {
          thrust::default_random_engine generator{};
          generator.discard(cuda::std::get<0>(tuple));
          cuda::std::get<1>(tuple).seed(generator());
        });
  }

  void initialize_particle_states_(const observation_type& initial_observation) noexcept {
    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), particle_states_.zip_begin()),
        thrust::make_zip_iterator(sampler_states_.end(),   particle_states_.zip_end()),
        [config = config_, initial_observation]
        PF_TARGET_ONLY_ATTRS(auto tuple) {
          sampler_type& sampler_state = cuda::std::get<0>(tuple);
          auto&         field_tuple   = cuda::std::get<1>(tuple);
          prediction_type::to_soa_tuple(field_tuple, config.sample_from(sampler_state, initial_observation));
        });

    most_likely_particle_state_ = reduce_most_likely_();
  }

  void initialize_particle_states_(
      const observation_type& initial_observation,
      const initialization_prior_param_type& initialization_prior) noexcept
    requires supports_initialization_prior_v {
    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), particle_states_.zip_begin()),
        thrust::make_zip_iterator(sampler_states_.end(),   particle_states_.zip_end()),
        [config = config_, initial_observation, initialization_prior]
        PF_TARGET_ONLY_ATTRS(auto tuple) {
          sampler_type& sampler_state = cuda::std::get<0>(tuple);
          auto&         field_tuple   = cuda::std::get<1>(tuple);
          prediction_type::to_soa_tuple(
              field_tuple,
              config.sample_from(sampler_state, initial_observation, initialization_prior));
        });

    most_likely_particle_state_ = reduce_most_likely_();
  }

  void initialize_internal_state_(const std::size_t& number_of_particles, const observation_type& initial_observation) noexcept {
    initialize_sampler_states_(number_of_particles);
    initialize_particle_states_(initial_observation);
  }

  void initialize_internal_state_(
      const std::size_t& number_of_particles,
      const observation_type& initial_observation,
      const initialization_prior_param_type& initialization_prior) noexcept
    requires supports_initialization_prior_v {
    initialize_sampler_states_(number_of_particles);
    initialize_particle_states_(initial_observation, initialization_prior);
  }

 public:
  template <typename... Ts>
  particle_filter(const std::size_t& number_of_particles, const observation_type& initial_observation, Ts&&... params) noexcept
      : caching_allocator_{helper::thread_local_caching_allocator()},
        config_(std::forward<Ts>(params)...),
        resampler_(number_of_particles),
        sampler_states_(number_of_particles),
        log_particle_weights_(number_of_particles),
        particle_states_(number_of_particles) {
    initialize_internal_state_(number_of_particles, initial_observation);
  }

  template <typename... Ts>
  particle_filter(
      const std::size_t& number_of_particles,
      const observation_type& initial_observation,
      const initialization_prior_param_type& initialization_prior,
      Ts&&... params) noexcept
      requires supports_initialization_prior_v
      : caching_allocator_{helper::thread_local_caching_allocator()},
        config_(std::forward<Ts>(params)...),
        resampler_(number_of_particles),
        sampler_states_(number_of_particles),
        log_particle_weights_(number_of_particles),
        particle_states_(number_of_particles) {
    initialize_internal_state_(number_of_particles, initial_observation, initialization_prior);
  }
};

}  // namespace pf::filter
