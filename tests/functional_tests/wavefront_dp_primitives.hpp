// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef _WAVE_DP_PRIMITIVE_TEST_HPP_
#define _WAVE_DP_PRIMITIVE_TEST_HPP_

#define SKIP_DONE_FLAG '2'
#define LOOP_DONE_FLAG '3'
#define DEST_RETURN_FLAG '5'

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
template <typename T>
class WaveDpPrimitiveTester : public Tester {
 public:
  explicit WaveDpPrimitiveTester(TesterArguments args);
  virtual ~WaveDpPrimitiveTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void verifyResults(size_t size) override;

  size_t buff_size = 0;
  int check_id = 1;

  T *source = nullptr;
  T *dest = nullptr;
  signed char *src_flags = nullptr; // for AMO tag
  signed char *dest_flags = nullptr; // for AMO tag
};

#endif
