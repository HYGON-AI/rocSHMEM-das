// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "team_alltoallv_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#define CHECK_ROCSHMEM(call)                                                  \
  do {                                                                        \
    if ((call) != ROCSHMEM_SUCCESS) {                                         \
      throw std::runtime_error("rocSHMEM call failed: " #call);              \
    }                                                                         \
  } while (0)

namespace {

unsigned char alltoallvByteValue(size_t sender, size_t receiver, size_t n_pes) {
  // Match memset's byte conversion without overflowing a rank product.
  return static_cast<unsigned char>((sender % 256) * (n_pes % 256) + receiver % 256);
}

size_t nonuniformCount(size_t nbytes, int n_pes, int sender) {
  const size_t total = nbytes * static_cast<size_t>(n_pes);
  const size_t weight_sum =
      static_cast<size_t>(n_pes) * (static_cast<size_t>(n_pes) + 1) / 2;
  const size_t weight = static_cast<size_t>(sender) + 1;
  size_t assigned = 0;
  for (size_t w = 1; w <= static_cast<size_t>(n_pes); ++w) {
    assigned += total * w / weight_sum;
  }
  const size_t remainder = total - assigned;
  return total * weight / weight_sum + (weight <= remainder ? 1 : 0);
}

}  // namespace

TeamAlltoallvOnStreamTester::TeamAlltoallvOnStreamTester(TesterArguments args)
    : Tester(args) {
  my_pe_ = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes_ = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  if (args.num_wgs <= 0 || n_pes_ <= 0 || args.skip < 0) {
    throw std::runtime_error("invalid stream, PE, or warmup count");
  }

  num_streams_ = static_cast<size_t>(args.num_wgs);
  const size_t pe_count = static_cast<size_t>(n_pes_);
  const size_t size_limit = std::numeric_limits<size_t>::max();

  // Check the largest allocation before multiplying sizes or forming offsets
  const size_t capacity = size_limit / num_streams_ / pe_count / 2;
  if (max_msg_size > capacity) {
    throw std::runtime_error("collective test buffer size overflows size_t");
  }

  const size_t max_bytes = max_msg_size;

  // Bound the total * weight multiplication in nonuniformCount.
  if (max_bytes > size_limit / pe_count / pe_count) {
    throw std::runtime_error("alltoallv weighted byte count overflows size_t");
  }

  bytes_per_stream_ = max_bytes * static_cast<size_t>(n_pes_);

  // Sender-weighted nonuniform data needs up to twice the source capacity
  source_bytes_per_stream_ = 2 * bytes_per_stream_;
  const size_t source_bytes = source_bytes_per_stream_ * num_streams_;
  const size_t dest_bytes = bytes_per_stream_ * num_streams_;

  source_buf_ = static_cast<char *>(alloc_test_buffer(source_bytes, args.local_buf_type));
  dest_buf_ = static_cast<char *>(alloc_test_buffer(dest_bytes));

  // Allocate metadata arrays for alltoallv
  if (num_streams_ > size_limit / pe_count / sizeof(size_t)) {
    throw std::runtime_error("alltoallv metadata size overflows size_t");
  }

  const size_t metadata_count = num_streams_ * pe_count;
  CHECK_HIP(hipMalloc(&dest_nelems_, metadata_count * sizeof(size_t)));
  CHECK_HIP(hipMalloc(&dest_displs_, metadata_count * sizeof(size_t)));
  CHECK_HIP(hipMalloc(&source_nelems_, metadata_count * sizeof(size_t)));
  CHECK_HIP(hipMalloc(&source_displs_, metadata_count * sizeof(size_t)));

  // Initialize streams and events
  teams_.resize(num_streams_, ROCSHMEM_TEAM_INVALID);
  team_ranks_.resize(num_streams_, -1);
  streams_.resize(num_streams_);
  starts_.resize(num_streams_);
  stops_.resize(num_streams_);

  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_HIP(hipStreamCreate(&streams_[i]));
    CHECK_HIP(hipEventCreate(&starts_[i]));
    CHECK_HIP(hipEventCreate(&stops_[i]));
  }
}

TeamAlltoallvOnStreamTester::~TeamAlltoallvOnStreamTester() {
  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_HIP(hipEventDestroy(stops_[i]));
    CHECK_HIP(hipEventDestroy(starts_[i]));
    CHECK_HIP(hipStreamDestroy(streams_[i]));
  }

  CHECK_HIP(hipFree(source_displs_));
  CHECK_HIP(hipFree(source_nelems_));
  CHECK_HIP(hipFree(dest_displs_));
  CHECK_HIP(hipFree(dest_nelems_));
  free_test_buffer(source_buf_, args.local_buf_type);
  free_test_buffer(dest_buf_);
}

void TeamAlltoallvOnStreamTester::preLaunchKernel() {
  // bw_factor belongs to the base tester and is still an int.
  if (n_pes_ > std::numeric_limits<int>::max() / n_pes_) {
    throw std::runtime_error("alltoallv bandwidth factor exceeds the base tester's int limit");
  }
  bw_factor = n_pes_ * n_pes_;
}

void TeamAlltoallvOnStreamTester::postLaunchKernel() {
  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_HIP(hipStreamSynchronize(streams_[i]));

    float elapsed_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, starts_[i], stops_[i]));

    if (i < num_timers) {
      start_time[i] = 0;
      end_time[i] = static_cast<long long>(elapsed_ms * wall_clk_rate);
    }
  }

  for (size_t i = num_streams_; i < num_timers; ++i) {
    start_time[i] = end_time[i] = 0;
  }

  for (size_t i = 0; i < num_streams_; ++i) {
    rocshmem_team_destroy(teams_[i]);
    teams_[i] = ROCSHMEM_TEAM_INVALID;
  }
}

void TeamAlltoallvOnStreamTester::resetBuffers(size_t size) {
  // Create teams and get ranks
  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_ROCSHMEM(rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 1, n_pes_, nullptr, 0, &teams_[i]));

    if (teams_[i] == ROCSHMEM_TEAM_INVALID) {
      throw std::runtime_error("failed to create duplicate team");
    }

    team_ranks_[i] = rocshmem_team_my_pe(teams_[i]);
  }

  if (size > max_msg_size) {
    throw std::runtime_error(
        "message size exceeds the allocated test capacity");
  }

  const size_t nbytes = size;
  const bool nonuniform_alltoallv =
      std::getenv("ROCSHMEM_TEST_ALLTOALLV_NONUNIFORM") != nullptr;
  const size_t used_per_stream = nbytes * n_pes_;

  // Initialize source and destination buffers
  for (size_t s = 0; s < num_streams_; ++s) {
    const int team_rank = team_ranks_[s];

    std::memset(dest_buf_ + s * bytes_per_stream_, 0,
                used_per_stream);

    char *src = source_buf_ + s * source_bytes_per_stream_;
    size_t source_offset = 0;

    for (int pe = 0; pe < n_pes_; ++pe) {
      const size_t send_count = nonuniform_alltoallv
                                   ? nonuniformCount(nbytes, n_pes_, team_rank)
                                   : nbytes;

      std::memset(src + source_offset,
                  alltoallvByteValue(team_rank, pe, n_pes_), send_count);
      source_offset += send_count;
    }
  }

  // Build metadata arrays for alltoallv
  std::vector<size_t> dest_counts(num_streams_ * n_pes_, nbytes);
  std::vector<size_t> source_counts(num_streams_ * n_pes_, nbytes);
  std::vector<size_t> dest_displs(num_streams_ * n_pes_);
  std::vector<size_t> source_displs(num_streams_ * n_pes_);

  for (size_t s = 0; s < num_streams_; ++s) {
    const int team_rank = team_ranks_[s];
    size_t source_offset = 0;
    size_t dest_offset = 0;

    for (int pe = 0; pe < n_pes_; ++pe) {
      const size_t index = s * n_pes_ + pe;

      const size_t send_count = nonuniform_alltoallv
                                    ? nonuniformCount(nbytes, n_pes_, team_rank)
                                    : nbytes;

      source_counts[index] = send_count;
      source_displs[index] = source_offset;
      source_offset += send_count;

      const size_t receive_count =
          nonuniform_alltoallv ? nonuniformCount(nbytes, n_pes_, pe) : nbytes;

      dest_counts[index] = receive_count;
      dest_displs[index] = dest_offset;
      dest_offset += receive_count;
    }
  }

  // Copy metadata to device
  const size_t meta_bytes = dest_counts.size() * sizeof(size_t);

  CHECK_HIP(hipMemcpy(dest_nelems_, dest_counts.data(), meta_bytes,
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(source_nelems_, source_counts.data(), meta_bytes,
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dest_displs_, dest_displs.data(), meta_bytes,
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(source_displs_, source_displs.data(), meta_bytes,
                      hipMemcpyHostToDevice));
}

void TeamAlltoallvOnStreamTester::enqueue(size_t s, [[maybe_unused]] size_t nbytes) {
  char *src = source_buf_ + s * source_bytes_per_stream_;
  char *dst = dest_buf_ + s * bytes_per_stream_;
  size_t *dn = dest_nelems_ + s * n_pes_;
  size_t *dd = dest_displs_ + s * n_pes_;
  size_t *sn = source_nelems_ + s * n_pes_;
  size_t *sd = source_displs_ + s * n_pes_;

  const int status = rocshmem_alltoallvmem_on_stream(
      teams_[s], dst, dn, dd, src, sn, sd, streams_[s]);

  CHECK_ROCSHMEM(status);
}

void TeamAlltoallvOnStreamTester::launchKernel(
    [[maybe_unused]] dim3 grid_size, [[maybe_unused]] dim3 block_size,
    int loop, size_t size) {

  // Validate loop counts to prevent overflow
  if (loop < 0 ||
      static_cast<size_t>(args.skip) >
          std::numeric_limits<size_t>::max() / num_streams_ ||
      static_cast<size_t>(loop) >
          std::numeric_limits<size_t>::max() / num_streams_ -
              static_cast<size_t>(args.skip)) {
    throw std::runtime_error("message count overflows size_t");
  }

  const size_t nbytes = size;

  // Warmup iterations
  for (int i = 0; i < args.skip; ++i) {
    for (size_t s = 0; s < num_streams_; ++s) {
      enqueue(s, nbytes);
    }
  }

  for (auto stream : streams_) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }

  // Timed iterations
  for (int i = 0; i < loop; ++i) {
    for (size_t s = 0; s < num_streams_; ++s) {
      if (i == 0) {
        CHECK_HIP(hipEventRecord(starts_[s], streams_[s]));
      }

      enqueue(s, nbytes);

      if (i == loop - 1) {
        CHECK_HIP(hipEventRecord(stops_[s], streams_[s]));
      }
    }
  }
  num_msgs = (static_cast<size_t>(loop) + static_cast<size_t>(args.skip)) * num_streams_;
  num_timed_msgs = static_cast<size_t>(loop) * num_streams_;
}

void TeamAlltoallvOnStreamTester::verifyResults(size_t size) {
  if (size > max_msg_size) {
    throw std::runtime_error("message size exceeds the allocated test capacity");
  }
  for (size_t s = 0; s < num_streams_; ++s) {
    const auto result = rocshmem::collective_verifier::verifyChar(
        dest_buf_ + s * bytes_per_stream_, size, n_pes_,
        dest_nelems_ + s * n_pes_, dest_displs_ + s * n_pes_,
        static_cast<size_t>(n_pes_), static_cast<size_t>(team_ranks_[s]),
        streams_[s]);
    if (result.failed) {
      std::cerr << "PE " << my_pe_ << " stream " << s << " index " << result.index
                << ": expected " << result.expected << ", got " << result.actual
                << std::endl;
      rocshmem_global_exit(1);
    }
  }
}

#undef CHECK_ROCSHMEM
