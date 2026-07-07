// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#include "sme_gaussian_blur_api.h"

#define WIDTH 20
#define HEIGHT 20

int main(void) {
  uint8_t src[WIDTH * HEIGHT];
  uint8_t dst[WIDTH * HEIGHT];

  // Input image with a vertical line in the middle.
  for (size_t y = 0; y < HEIGHT; ++y) {
    for (size_t x = 0; x < WIDTH; ++x) {
      src[x + y * WIDTH] = (x == (WIDTH / 2)) ? 255 : 0;
    }
  }

  // Execute 15x15 GaussianBlur with default sigma.
  // As src is single channel uint8_t stride is equal to WIDTH.
  sme_gaussian_blur_u8(src, WIDTH, dst, WIDTH, WIDTH, HEIGHT, 1, 15, 15, 0.0F,
                       0.0F, KLEIDICV_BORDER_TYPE_REFLECT);

  // Print raw pixel values to show that the middle vertical line was blurred.
  printf("Raw pixel values for the blurred output:\n");
  for (size_t y = 0; y < HEIGHT; ++y) {
    for (size_t x = 0; x < WIDTH; ++x) {
      printf("%d\t", dst[x + y * WIDTH]);
    }
    putchar('\n');
  }

  return 0;
}
