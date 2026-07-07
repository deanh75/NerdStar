// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MORPHOLOGY_MORPHOLOGY_H
#define KLEIDICV_MORPHOLOGY_MORPHOLOGY_H

#include "kleidicv/ctypes.h"
#include "kleidicv/morphology/workspace.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

inline bool morphology_is_implemented(size_t width, size_t height,
                                      size_t kernel_width, size_t kernel_height,
                                      size_t channels) KLEIDICV_STREAMING {
  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return false;
  }

  if (width < kernel_width - 1 || height < kernel_height - 1) {
    return false;
  }

  return true;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MORPHOLOGY_MORPHOLOGY_H
