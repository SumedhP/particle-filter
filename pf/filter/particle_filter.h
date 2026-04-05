#pragma once

#include <pf/config/target_config.h>
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
#include <concepts>
#include <cstdint>
#include <cuda/std/tuple>
#include <limits>
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

template <typename ParticleFilterConfiguration>
concept supports_likelihood_precompute = requires(
    const ParticleFilterConfiguration& config,
    const typename ParticleFilterConfiguration::observation_type& observation,
    const typename ParticleFilterConfiguration::sampler_type& sampler,
    const typename ParticleFilterConfiguration::prediction_type& prediction) {
  { config.precompute_likelihood_evaluation_context(observation) };
  {
    config.conditional_log_likelihood_from_precomputed(
        sampler,
        config.precompute_likelihood_evaluation_context(observation),
        prediction)
  } -> std::convertible_to<float>;
};

template <typename ParticleFilterConfiguration>
concept supports_rough_likelihood_precompute = requires(
    const ParticleFilterConfiguration& config,
    const typename ParticleFilterConfiguration::observation_type& observation,
    const typename ParticleFilterConfiguration::sampler_type& sampler,
    const typename ParticleFilterConfiguration::prediction_type& prediction) {
  {
    config.rough_conditional_log_likelihood_from_precomputed(
        sampler,
        config.precompute_likelihood_evaluation_context(observation),
        prediction)
  } -> std::convertible_to<float>;
  { config.refinement_log_likelihood_window() } -> std::convertible_to<float>;
};

template <typename ParticleFilterConfiguration>
concept supports_observation_resample_period = requires(const ParticleFilterConfiguration& config) {
  { config.observation_resample_period() } -> std::convertible_to<std::uint32_t>;
};

template <typename ParticleFilterConfiguration>
concept supports_observation_update_subsample_stride = requires(const ParticleFilterConfiguration& config) {
  { config.observation_update_subsample_stride() } -> std::convertible_to<std::uint32_t>;
};

template <typename ParticleFilterConfiguration>
concept supports_initial_sampling_precompute = requires(
    const ParticleFilterConfiguration& config,
    const typename ParticleFilterConfiguration::observation_type& observation,
    typename ParticleFilterConfiguration::sampler_type& sampler) {
  { config.precompute_initial_sampling_context(observation) };
  {
    config.sample_from_precomputed(
        sampler,
        config.precompute_initial_sampling_context(observation))
  } -> std::convertible_to<typename ParticleFilterConfiguration::prediction_type>;
};

PF_TARGET_ATTRS [[nodiscard]] inline std::uint64_t splitmix64(const std::uint64_t& value) noexcept {
  std::uint64_t z = value + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace helper

template <typename ParticleFilterConfiguration>
  requires concepts::particle_filter_configuration<ParticleFilterConfiguration>
class particle_filter {
 private:
  using particle_configuration_type = ParticleFilterConfiguration;

  using observation_type = typename ParticleFilterConfiguration::observation_type;
  using prediction_type = typename ParticleFilterConfiguration::prediction_type;
  using sampler_type = typename ParticleFilterConfiguration::sampler_type;
  using sampler_seed_type = typename sampler_type::random_number_generator_type::result_type;

  mutable helper::caching_allocator_type caching_allocator_;
  ParticleFilterConfiguration config_;

  mutable prediction_type most_likely_particle_state_;
  mutable bool most_likely_particle_state_ready_;
  mutable bool particle_states_ready_;
  observation_type initial_observation_;
  std::uint64_t observation_update_counter_;
  systematic_resampler<prediction_type, std::uint32_t> resampler_;

  mutable target_config::vector<sampler_type> sampler_states_;
  target_config::vector<float> log_particle_weights_;
  mutable target_config::vector<prediction_type> particle_states_;

  void ensure_log_particle_weights_ready_() noexcept {
    if (log_particle_weights_.size() != particle_states_.size()) {
      log_particle_weights_.resize(particle_states_.size());
    }
  }

 public:
  void ensure_particle_states_ready_() const noexcept {
    if (particle_states_ready_) {
      return;
    }

    if constexpr (helper::supports_initial_sampling_precompute<particle_configuration_type>) {
      const auto initial_sampling_context = config_.precompute_initial_sampling_context(initial_observation_);

      thrust::transform(
          target_config::policy(caching_allocator_),
          sampler_states_.begin(),
          sampler_states_.end(),
          particle_states_.begin(),
          [config = config_, initial_sampling_context] PF_TARGET_ONLY_ATTRS(sampler_type & sampler_state) {
            return config.sample_from_precomputed(sampler_state, initial_sampling_context);
          });
    } else {
      thrust::transform(
          target_config::policy(caching_allocator_),
          sampler_states_.begin(),
          sampler_states_.end(),
          particle_states_.begin(),
          [config = config_, initial_observation = initial_observation_] PF_TARGET_ONLY_ATTRS(sampler_type & sampler_state) {
            return config.sample_from(sampler_state, initial_observation);
          });
    }

    particle_states_ready_ = true;
    most_likely_particle_state_ready_ = false;
  }

 private:

  void refresh_most_likely_particle_state_() const noexcept {
    ensure_particle_states_ready_();

    if (most_likely_particle_state_ready_) {
      return;
    }

    most_likely_particle_state_ = thrust::transform_reduce(
                                      target_config::policy(caching_allocator_),
                                      particle_states_.cbegin(),
                                      particle_states_.cend(),
                                      particle_reduction_state_transform<prediction_type>(),
                                      particle_reduction_state<prediction_type>::zero(),
                                      config_.most_likely_particle_reduction())
                                      .most_likely_particle();
    most_likely_particle_state_ready_ = true;
  }

 public:
  [[nodiscard]] prediction_type extrapolate_state(const float& time_offset_seconds) const noexcept {
    refresh_most_likely_particle_state_();
    return most_likely_particle_state_.extrapolate_state(time_offset_seconds);
  }

  void update_state_sans_observation(const float& time_offset_seconds) noexcept {
    ensure_particle_states_ready_();

    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), particle_states_.begin()),
        thrust::make_zip_iterator(sampler_states_.end(), particle_states_.end()),
        [config = config_, time_offset_seconds] PF_TARGET_ONLY_ATTRS(cuda::std::tuple<sampler_type&, prediction_type&> tuple) {
          sampler_type& sampler_state = cuda::std::get<0>(tuple);
          prediction_type& particle_state = cuda::std::get<1>(tuple);
          config.apply_process(time_offset_seconds, sampler_state, particle_state);
        });
    most_likely_particle_state_ready_ = false;
  }

  void update_state_with_observation(const float& time_offset_seconds, const observation_type& observation_state) noexcept {
    ensure_particle_states_ready_();
    ensure_log_particle_weights_ready_();

    observation_update_counter_ += 1U;

    std::uint32_t observation_update_subsample_stride = 1U;
    std::uint32_t observation_update_subsample_phase = 0U;

    if constexpr (helper::supports_observation_update_subsample_stride<particle_configuration_type>) {
      observation_update_subsample_stride = std::max<std::uint32_t>(1U, config_.observation_update_subsample_stride());
      observation_update_subsample_phase =
          static_cast<std::uint32_t>(observation_update_counter_ % observation_update_subsample_stride);
    }

    thrust::counting_iterator<std::size_t> index_sequence_begin(std::size_t{});
    constexpr float skipped_particle_penalty = 0.25f;

    if constexpr (helper::supports_likelihood_precompute<particle_configuration_type>) {
      const auto likelihood_context = config_.precompute_likelihood_evaluation_context(observation_state);

      if constexpr (helper::supports_rough_likelihood_precompute<particle_configuration_type>) {
        thrust::for_each(
            target_config::policy(caching_allocator_),
            thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.begin()),
            thrust::make_zip_iterator(
                index_sequence_begin + sampler_states_.size(),
                sampler_states_.end(),
                log_particle_weights_.end(),
                particle_states_.end()),
            [config = config_,
             time_offset_seconds,
             skipped_particle_penalty,
             observation_update_subsample_stride,
             observation_update_subsample_phase,
             likelihood_context] PF_TARGET_ONLY_ATTRS(cuda::std::tuple<std::size_t, sampler_type&, float&, prediction_type&> tuple) {
              const std::size_t particle_index = cuda::std::get<0>(tuple);
              sampler_type& sampler_state = cuda::std::get<1>(tuple);
              float& particle_weight = cuda::std::get<2>(tuple);
              prediction_type& particle_state = cuda::std::get<3>(tuple);

              if ((particle_index % observation_update_subsample_stride) != observation_update_subsample_phase) {
                particle_weight -= skipped_particle_penalty;
                return;
              }

              config.apply_process(time_offset_seconds, sampler_state, particle_state);
              particle_weight = config.rough_conditional_log_likelihood_from_precomputed(
                  sampler_state,
                  likelihood_context,
                  particle_state);
            });

        const float refinement_window = config_.refinement_log_likelihood_window();

        if (refinement_window > 0.0f) {
              const float rough_max = thrust::reduce(
                target_config::policy(caching_allocator_),
                log_particle_weights_.cbegin(),
                log_particle_weights_.cend(),
                std::numeric_limits<float>::lowest(),
                thrust::maximum<float>{});

          constexpr float non_refined_penalty = 1.5f;

          thrust::for_each(
              target_config::policy(caching_allocator_),
              thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.begin()),
              thrust::make_zip_iterator(
                  index_sequence_begin + sampler_states_.size(),
                  sampler_states_.end(),
                  log_particle_weights_.end(),
                  particle_states_.end()),
              [config = config_,
               rough_max,
               refinement_window,
               observation_update_subsample_stride,
               observation_update_subsample_phase,
               likelihood_context] PF_TARGET_ONLY_ATTRS(cuda::std::tuple<std::size_t, sampler_type&, float&, prediction_type&> tuple) {
                const std::size_t particle_index = cuda::std::get<0>(tuple);
                sampler_type& sampler_state = cuda::std::get<1>(tuple);
                float& particle_weight = cuda::std::get<2>(tuple);
                prediction_type& particle_state = cuda::std::get<3>(tuple);

                if ((particle_index % observation_update_subsample_stride) != observation_update_subsample_phase) {
                  return;
                }

                if (particle_weight >= rough_max - refinement_window) {
                  particle_weight = config.conditional_log_likelihood_from_precomputed(
                      sampler_state,
                      likelihood_context,
                      particle_state);
                } else {
                  particle_weight -= non_refined_penalty;
                }
              });
        } else if (refinement_window == 0.0f) {
          thrust::for_each(
              target_config::policy(caching_allocator_),
              thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.begin()),
              thrust::make_zip_iterator(
                  index_sequence_begin + sampler_states_.size(),
                  sampler_states_.end(),
                  log_particle_weights_.end(),
                  particle_states_.end()),
              [config = config_,
               skipped_particle_penalty,
               observation_update_subsample_stride,
               observation_update_subsample_phase,
               likelihood_context] PF_TARGET_ONLY_ATTRS(cuda::std::tuple<std::size_t, sampler_type&, float&, prediction_type&> tuple) {
                const std::size_t particle_index = cuda::std::get<0>(tuple);
                sampler_type& sampler_state = cuda::std::get<1>(tuple);
                float& particle_weight = cuda::std::get<2>(tuple);
                prediction_type& particle_state = cuda::std::get<3>(tuple);

                if ((particle_index % observation_update_subsample_stride) != observation_update_subsample_phase) {
                  particle_weight -= skipped_particle_penalty;
                  return;
                }

                particle_weight = config.conditional_log_likelihood_from_precomputed(
                    sampler_state,
                    likelihood_context,
                    particle_state);
              });
        } else {
          // Negative refinement window enables rough-only likelihood mode.
        }
      } else {
        thrust::for_each(
            target_config::policy(caching_allocator_),
            thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.begin()),
            thrust::make_zip_iterator(
                index_sequence_begin + sampler_states_.size(),
                sampler_states_.end(),
                log_particle_weights_.end(),
                particle_states_.end()),
            [config = config_,
             time_offset_seconds,
             skipped_particle_penalty,
             observation_update_subsample_stride,
             observation_update_subsample_phase,
             likelihood_context] PF_TARGET_ONLY_ATTRS(cuda::std::tuple<std::size_t, sampler_type&, float&, prediction_type&> tuple) {
              const std::size_t particle_index = cuda::std::get<0>(tuple);
              sampler_type& sampler_state = cuda::std::get<1>(tuple);
              float& particle_weight = cuda::std::get<2>(tuple);
              prediction_type& particle_state = cuda::std::get<3>(tuple);

              if ((particle_index % observation_update_subsample_stride) != observation_update_subsample_phase) {
                particle_weight -= skipped_particle_penalty;
                return;
              }

              config.apply_process(time_offset_seconds, sampler_state, particle_state);
              particle_weight = config.conditional_log_likelihood_from_precomputed(
                  sampler_state,
                  likelihood_context,
                  particle_state);
            });
      }
    } else {
      thrust::for_each(
          target_config::policy(caching_allocator_),
          thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.begin()),
          thrust::make_zip_iterator(
              index_sequence_begin + sampler_states_.size(),
              sampler_states_.end(),
              log_particle_weights_.end(),
              particle_states_.end()),
          [config = config_,
           time_offset_seconds,
           skipped_particle_penalty,
           observation_update_subsample_stride,
           observation_update_subsample_phase,
           observation_state] PF_TARGET_ONLY_ATTRS(cuda::std::tuple<std::size_t, sampler_type&, float&, prediction_type&> tuple) {
            const std::size_t particle_index = cuda::std::get<0>(tuple);
            sampler_type& sampler_state = cuda::std::get<1>(tuple);
            float& particle_weight = cuda::std::get<2>(tuple);
            prediction_type& particle_state = cuda::std::get<3>(tuple);

            if ((particle_index % observation_update_subsample_stride) != observation_update_subsample_phase) {
              particle_weight -= skipped_particle_penalty;
              return;
            }

            config.apply_process(time_offset_seconds, sampler_state, particle_state);
            particle_weight = config.conditional_log_likelihood(sampler_state, observation_state, particle_state);
          });
    }

    bool should_resample = true;

    if constexpr (helper::supports_observation_resample_period<particle_configuration_type>) {
      const std::uint32_t period = std::max<std::uint32_t>(1U, config_.observation_resample_period());
      should_resample = (observation_update_counter_ % period) == 0U;
    }

    if (should_resample) {
      resampler_.resample(target_config::policy(caching_allocator_), log_particle_weights_, particle_states_);
    }

    most_likely_particle_state_ready_ = false;
  }

  void initialize_internal_state_(const std::size_t& number_of_particles, const observation_type& initial_observation) noexcept {
    (void)initial_observation;

    thrust::counting_iterator<std::size_t> index_sequence_begin(std::size_t{});

    thrust::for_each(
        target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin()),
        thrust::make_zip_iterator(index_sequence_begin + number_of_particles, sampler_states_.end()),
        [] PF_TARGET_ATTRS(cuda::std::tuple<std::size_t, sampler_type&> tuple) {
          const std::size_t index = cuda::std::get<0>(tuple);
          sampler_type& sampler_state = cuda::std::get<1>(tuple);

          const std::uint64_t mixed = helper::splitmix64(static_cast<std::uint64_t>(index));
          sampler_seed_type seed = static_cast<sampler_seed_type>(mixed);

          if constexpr (sizeof(sampler_seed_type) < sizeof(std::uint64_t)) {
            seed ^= static_cast<sampler_seed_type>(mixed >> (8 * sizeof(sampler_seed_type)));
          }

          sampler_state.seed(seed);
        });

    particle_states_ready_ = false;
    most_likely_particle_state_ready_ = false;
  }

  template <typename... Ts>
  particle_filter(const std::size_t& number_of_particles, const observation_type& initial_observation, Ts&&... params) noexcept
      : caching_allocator_{helper::thread_local_caching_allocator()},
        config_(std::forward<Ts>(params)...),
        most_likely_particle_state_{},
        most_likely_particle_state_ready_{false},
        particle_states_ready_{false},
        initial_observation_{initial_observation},
        observation_update_counter_{0},
        resampler_(number_of_particles),
        sampler_states_(number_of_particles),
        log_particle_weights_(),
        particle_states_(number_of_particles) {
    initialize_internal_state_(number_of_particles, initial_observation);
  }
};

}  // namespace pf::filter
