// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#if __has_include(<getopt.h>)
#include <getopt.h>
#endif  // has include getopt.h

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <string>

#include "framework/utils.h"

namespace test {

// Default vector length expects Neon (128 bits).
size_t Options::vector_length_{16};
uint64_t Options::seed_{0};

#if __has_include(<getopt.h>)
// Parses command line arguments.
static void parse_arguments(int argc, char **argv, bool &is_seed_set) {
  // clang-format off
  static struct option long_options[] = {
    {"vector-length", required_argument, nullptr, 'v'},
    {"seed", required_argument, nullptr, 's'},
    {"long-running-tests", no_argument, nullptr, 'l'},
    {nullptr, 0, nullptr, 0}
  };
  // clang-format on

  for (;;) {
    int option_index = 0;
    int opt = getopt_long(argc, argv, "", long_options, &option_index);
    if (opt == -1) {
      break;
    }

    switch (opt) {
      default:
        std::cerr << "Unknown command line argument." << std::endl;
        break;

      case 'v':
        Options::set_vector_length(std::stoi(optarg));
        break;

      case 's':
        Options::set_seed(std::stoull(optarg));
        is_seed_set = true;
        break;

      case 'l':
        Options::turn_on_long_running_tests();
        break;
    }
  }
}
#endif
}  // namespace test

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  bool is_seed_set = false;

#if __has_include(<getopt.h>)
  test::parse_arguments(argc, argv, is_seed_set);
#endif  // has include getopt.h

  // Set a random seed if it is not explicitly specified.
  if (!is_seed_set) {
    std::random_device rd;
    test::Options::set_seed(static_cast<uint64_t>(rd()));
  }

  std::cout << "Vector length is set to " << test::Options::vector_length()
            << " bytes." << std::endl;
  std::cout << "Seed is set to " << test::Options::seed() << "." << std::endl;
  return RUN_ALL_TESTS();
}
