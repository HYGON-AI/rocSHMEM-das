// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef _AMO_DP_STANDARD_TESTER_HPP_
#define _AMO_DP_STANDARD_TESTER_HPP_

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
template <typename T>
class AMODpStandardTester : public Tester {
 public:
  explicit AMODpStandardTester(TesterArguments args);
  virtual ~AMODpStandardTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void verifyResults(size_t size) override;

  void verifyDestValues();
  void verifyReturnValues();

  int  destIndex(int l, int elem_idx) const;
  int  numElems() const;
  std::pair<T*, int> retChunk(int l, int elem_idx) const;
  int check_id = 1;

  T* dest{nullptr};        // symmetric target buffer [loop][elem]
  T* ret_val{nullptr};     // device returns [loop][thread]
  long long* done_flags{nullptr}; // for AMO tags

  size_t n_in{0};          // num_wgs * wg_size
  size_t n_out{0};         // elements per loop: PerBlock->num_wgs, PerGrid->1
  size_t n_loops{0};       // loop + skip
};

#endif
