// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>

#include "kleidicv/kleidicv.h"

int main(void) {
  uint8_t a[4] = {1, 2, 3, 4};
  uint8_t b[4] = {5, 6, 7, 8};
  uint8_t c[4] = {0, 0, 0, 0};

  kleidicv_saturating_add_u8(a, sizeof(a[0]), b, sizeof(b[0]), c, sizeof(c[0]),
                             4, 1);

  for (size_t i = 0; i < (sizeof(a) / sizeof(a[0])); i++) {
    printf("%hhu + %hhu = %hhu\n", a[i], b[i], c[i]);
  }

  return 0;
}
