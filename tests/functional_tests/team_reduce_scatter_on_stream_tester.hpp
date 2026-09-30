// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TEAM_REDUCE_SCATTER_ON_STREAM_TESTER_HPP_
#define TEAM_REDUCE_SCATTER_ON_STREAM_TESTER_HPP_

#include <hip/hip_runtime.h>
#include <vector>

#include "tester.hpp"

class TeamReduceScatterOnStreamTester : public Tester {
 public:
  explicit TeamReduceScatterOnStreamTester(TesterArguments args);
  ~TeamReduceScatterOnStreamTester() override;

 protected:
  void resetBuffers(size_t size) override;
  void preLaunchKernel() override;
  void launchKernel(dim3 grid_size, dim3 block_size, int loop, size_t size) override;
  void postLaunchKernel() override;
  void verifyResults(size_t size) override;

 private:
  void enqueue(size_t stream_id, size_t nelems);
  int *source_buf_{};
  int *dest_buf_{};
  int my_pe_{};
  int n_pes_{};
  size_t num_streams_{};
  size_t elements_per_stream_{};
  size_t source_elements_per_stream_{};
  std::vector<rocshmem_team_t> teams_;
  std::vector<hipStream_t> streams_;
  std::vector<hipEvent_t> starts_;
  std::vector<hipEvent_t> stops_;
};

#include "team_reduce_scatter_on_stream_tester.cpp"
#endif
