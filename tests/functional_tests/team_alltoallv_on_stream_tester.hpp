// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TEAM_ALLTOALLV_ON_STREAM_TESTER_HPP_
#define TEAM_ALLTOALLV_ON_STREAM_TESTER_HPP_

#include <hip/hip_runtime.h>
#include <vector>

#include "tester.hpp"

class TeamAlltoallvOnStreamTester : public Tester {
 public:
  explicit TeamAlltoallvOnStreamTester(TesterArguments args);
  ~TeamAlltoallvOnStreamTester() override;

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
  size_t *dest_nelems_{};
  size_t *dest_displs_{};
  size_t *source_nelems_{};
  size_t *source_displs_{};
  int my_pe_{};
  int n_pes_{};
  size_t num_streams_{};
  size_t bytes_per_stream_{};
  size_t source_bytes_per_stream_{};
  std::vector<rocshmem_team_t> teams_;
  std::vector<int> team_ranks_;
  std::vector<hipStream_t> streams_;
  std::vector<hipEvent_t> starts_;
  std::vector<hipEvent_t> stops_;
};

#include "team_alltoallv_on_stream_tester.cpp"
#endif
