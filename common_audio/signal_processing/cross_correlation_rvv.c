/*
 *  Copyright (c) 2026 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "common_audio/signal_processing/cross_correlation.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// RVV (VLEN=256) version of WebRtcSpl_CrossCorrelationC().
//
// Each term is rounded (arithmetic right shift) BEFORE accumulation to stay
// bit-exact with the C reference. Using vwmul (i16xi16 -> i32) + vsra on each
// product then a single vector accumulator per lag matches the scalar result.
// LMUL=2 (32 int16 lanes / 32 int32 lanes) is the throughput sweet spot on
// SpacemiT X100: larger LMUL spills vector registers during vwmul.
void WebRtcSpl_CrossCorrelationRvv(int32_t* cross_correlation,
                                   const int16_t* seq1,
                                   const int16_t* seq2,
                                   size_t dim_seq,
                                   size_t dim_cross_correlation,
                                   int right_shifts,
                                   int step_seq2) {
  for (size_t i = 0; i < dim_cross_correlation; i++) {
    const int16_t* a = seq1;
    const int16_t* b = seq2 + step_seq2 * (int)i;
    size_t n = dim_seq;

    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, 32);

    while (n >= 32) {
      vint16m2_t va = __riscv_vle16_v_i16m2(a, 32);
      vint16m2_t vb = __riscv_vle16_v_i16m2(b, 32);
      vint32m4_t prod = __riscv_vwmul_vv_i32m4(va, vb, 32);
      prod = __riscv_vsra_vx_i32m4(prod, right_shifts, 32);
      acc = __riscv_vadd_vv_i32m4(acc, prod, 32);
      a += 32;
      b += 32;
      n -= 32;
    }
    while (n != 0) {
      const size_t vl = __riscv_vsetvl_e16m2(n);
      vint16m2_t va = __riscv_vle16_v_i16m2(a, vl);
      vint16m2_t vb = __riscv_vle16_v_i16m2(b, vl);
      vint32m4_t prod = __riscv_vwmul_vv_i32m4(va, vb, vl);
      prod = __riscv_vsra_vx_i32m4(prod, right_shifts, vl);
      acc = __riscv_vadd_vv_i32m4(acc, prod, vl);
      a += vl;
      b += vl;
      n -= vl;
    }

    vint32m1_t reduced = __riscv_vredsum_vs_i32m4_i32m1(
        acc, __riscv_vmv_v_x_i32m1(0, 8), 32);
    *cross_correlation++ = __riscv_vmv_x_s_i32m1_i32(reduced);
  }
}
