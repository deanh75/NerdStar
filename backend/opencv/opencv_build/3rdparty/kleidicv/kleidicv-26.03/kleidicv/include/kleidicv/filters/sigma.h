// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SIGMA_H
#define KLEIDICV_SIGMA_H

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "kleidicv/config.h"

namespace KLEIDICV_TARGET_NAMESPACE {

static constexpr size_t get_half_kernel_size(size_t kernel_size)
    KLEIDICV_STREAMING {
  // since kernel sizes are odd, "half" here means that
  // the extra element is included
  return (kernel_size >> 1) + 1;
}

// This function is not marked as streaming compatible, as std::round is also
// not streaming compatible. Returns false if all the kernel's energy is
// concentrated in the mid element, in that case the filtering is equivalent to
// copying the input image.
static bool generate_gaussian_half_kernel(uint8_t* half_kernel,
                                          size_t half_size, float sigma) {
  // Define the mid point of the full kernel range.
  const size_t kMid = half_size - 1;

  // Define the full kernel size.
  const size_t kKernelSize = kMid * 2 + 1;

  // Calculate the sigma manually in case it is not defined.
  if (sigma == 0.0) {
    sigma = static_cast<float>(kKernelSize) * 0.15F + 0.35F;
  }

  // Temporary float half-kernel.
  float half_kernel_float[255];

  // Prepare the sigma value for later multiplication inside a loop.
  float coefficient = 1 / -(2 * sigma * sigma);

  float sum = 0.0;
  size_t j = kMid;
  for (size_t i = 0; i < kMid; i++, j--) {
    half_kernel_float[i] =
        std::exp(static_cast<float>(j) * static_cast<float>(j) * coefficient);
    sum += half_kernel_float[i];
  }

  // This multiplier serves two purposes:
  // * Normalizes the kernel values, so the sum of the final values is 1.
  //   (The 'sum' variable only accounts for the half of the kernel values
  //   without the mid point. This is the reason for the division by
  //   '(sum * 2 + 1)'.)
  // * Converts the values to fixed-point (uint8_t), where all the bits are used
  //   for the fractional part. (Result is less than 1.0.) This is the
  //   reason for the multiplication by 256.
  float multiplier = 256 / (sum * 2 + 1);

  // Normalize the kernel and convert it to fixed-point format. Rounding errors
  // are diffused in the kernel.
  float error = 0.0;
  for (size_t i = 0; i < kMid; i++) {
    float value = half_kernel_float[i] * multiplier - error;
    float value_rounded = std::round(value);
    half_kernel[i] = static_cast<uint8_t>(value_rounded);
    error = value_rounded - value;
  }
  uint16_t mid_value = static_cast<uint16_t>(std::round(multiplier - error));
  if (mid_value == 256) {
    return false;
  }
  half_kernel[kMid] = static_cast<uint8_t>(mid_value);
  return true;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SIGMA_H
