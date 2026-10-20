// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TEAM_FCOLLECT_ON_STREAM_TESTER_HPP_
#define TEAM_FCOLLECT_ON_STREAM_TESTER_HPP_

#include <hip/hip_runtime.h>
#include <vector>

#include "tester.hpp"

class TeamFcollectOnStreamTester : public Tester {
 public:
  explicit TeamFcollectOnStreamTester(TesterArguments args);
  ~TeamFcollectOnStreamTester() override;

 protected:
  void resetBuffers(size_t size) override;
  void preLaunchKernel() override;
  void launchKernel(dim3 grid_size, dim3 block_size, int loop, size_t size) override;
  void postLaunchKernel() override;
  void verifyResults(size_t size) override;

 private:
  void enqueue(size_t stream_id, size_t nbytes);
  char *source_buf_{};
  char *dest_buf_{};
  int my_pe_{};
  int n_pes_{};
  size_t num_streams_{};
  size_t bytes_per_stream_{};
  size_t source_bytes_per_stream_{};
  std::vector<rocshmem_team_t> teams_;
  std::vector<hipStream_t> streams_;
  std::vector<hipEvent_t> starts_;
  std::vector<hipEvent_t> stops_;
};

#include "team_fcollect_on_stream_tester.cpp"
#endif
