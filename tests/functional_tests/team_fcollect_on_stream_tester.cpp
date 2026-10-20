// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "team_fcollect_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#define CHECK_ROCSHMEM(call)                                                  \
  do {                                                                        \
    if ((call) != ROCSHMEM_SUCCESS) {                                         \
      throw std::runtime_error("rocSHMEM call failed: " #call);              \
    }                                                                         \
  } while (0)

TeamFcollectOnStreamTester::TeamFcollectOnStreamTester(TesterArguments args)
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
  const size_t capacity = size_limit / num_streams_ / pe_count;

  if (max_msg_size > capacity) {
    throw std::runtime_error("collective test buffer size overflows size_t");
  }

  const size_t max_bytes = max_msg_size;
  bytes_per_stream_ = max_bytes * static_cast<size_t>(n_pes_);
  source_bytes_per_stream_ = max_bytes;

  const size_t source_bytes = source_bytes_per_stream_ * num_streams_;
  const size_t dest_bytes = bytes_per_stream_ * num_streams_;

  source_buf_ = static_cast<char *>(alloc_test_buffer(source_bytes, args.local_buf_type));
  dest_buf_ = static_cast<char *>(alloc_test_buffer(dest_bytes));

  // Initialize streams and events
  teams_.resize(num_streams_, ROCSHMEM_TEAM_INVALID);
  streams_.resize(num_streams_);
  starts_.resize(num_streams_);
  stops_.resize(num_streams_);

  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_HIP(hipStreamCreate(&streams_[i]));
    CHECK_HIP(hipEventCreate(&starts_[i]));
    CHECK_HIP(hipEventCreate(&stops_[i]));
  }
}

TeamFcollectOnStreamTester::~TeamFcollectOnStreamTester() {
  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_HIP(hipEventDestroy(stops_[i]));
    CHECK_HIP(hipEventDestroy(starts_[i]));
    CHECK_HIP(hipStreamDestroy(streams_[i]));
  }

  free_test_buffer(source_buf_, args.local_buf_type);
  free_test_buffer(dest_buf_);
}

void TeamFcollectOnStreamTester::preLaunchKernel() {
  bw_factor = n_pes_;
}

void TeamFcollectOnStreamTester::postLaunchKernel() {
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

void TeamFcollectOnStreamTester::resetBuffers(size_t size) {
  // Create teams for each stream
  for (size_t i = 0; i < num_streams_; ++i) {
    CHECK_ROCSHMEM(rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 1, n_pes_, nullptr, 0, &teams_[i]));

    if (teams_[i] == ROCSHMEM_TEAM_INVALID) {
      throw std::runtime_error("failed to create duplicate team");
    }
  }

  if (size > max_msg_size) {
    throw std::runtime_error(
        "message size exceeds the allocated test capacity");
  }

  const size_t nbytes = size;
  const size_t used_per_stream = nbytes * n_pes_;

  // Initialize source and destination buffers
  for (size_t s = 0; s < num_streams_; ++s) {
    std::memset(dest_buf_ + s * bytes_per_stream_, 0,
                used_per_stream);

    char *src = source_buf_ + s * source_bytes_per_stream_;
    const unsigned char value = static_cast<unsigned char>(static_cast<size_t>(my_pe_) + 1);
    std::memset(src, value, nbytes);
  }
}

void TeamFcollectOnStreamTester::enqueue(size_t s, size_t nbytes) {
  char *src = source_buf_ + s * source_bytes_per_stream_;
  char *dst = dest_buf_ + s * bytes_per_stream_;

  const int status = rocshmem_fcollectmem_on_stream(
      teams_[s], dst, src, nbytes, streams_[s]);

  CHECK_ROCSHMEM(status);
}

void TeamFcollectOnStreamTester::launchKernel(
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

void TeamFcollectOnStreamTester::verifyResults(size_t size) {
  if (size > max_msg_size) {
    throw std::runtime_error("message size exceeds the allocated test capacity");
  }
  for (size_t s = 0; s < num_streams_; ++s) {
    const auto result = rocshmem::collective_verifier::verifyChar(
        dest_buf_ + s * bytes_per_stream_, size, n_pes_,
        nullptr, nullptr, 1, 1,
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
