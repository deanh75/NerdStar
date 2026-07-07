<!--
SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Contributing to Kleidicv

## Licensing

Use of KleidiCV is subject to an Apache-2.0 license. The license can be found at
`LICENSES/Apache-2.0.txt`. We will receive inbound contributions under the same
license.

## How to submit a patch

Contributions are managed on [Arm's Gitlab](https://gitlab.arm.com) via merge
requests. To create a merge request first the project needs to be forked which
requires permission. Details on how to request this permission are given
[here](https://gitlab.arm.com/documentation/contributions).

## Code style

KleidiCV follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
with the following exceptions:
* Function names are `snake_case`.
* Where the
  [Arm C Language Extensions](https://arm-software.github.io/acle/main/acle.html)
  mandates nonstandard C++ extensions, this takes precedence over the style
  guide.

The formatting is checked with `clang-format` before accepting a change. Source
can be formatted locally with `scripts/format.sh`. (It is the developerss
responsibility to provide the `clang-format` binary. Please use
`clang-format-20`.)

## No dependency on C++ standard library at runtime

KleidiCV must not have any dependencies on the C++ runtime library to enable its
usage on systems without a C++ runtime. This means that only compile-time C++
constructs can be used in KleidiCV. However, there is not such a restriction for
the test and benchmark sources.

## Documentation

KlieidiCV's public API is documented in `kleidicv/include/kleidicv/kleidicv.h`
with Doxygen comments in Javadoc style.

## Testing

Each function with a prototype in `kleidicv.h` must be accompanied by a set of
tests that run the function on a set of representative inputs and compare the
result to known-good output. These tests are placed in the `test/api` folder
(see [Testing](doc/test.md)) and must exercise every path in the function under
test.

## Benchmarks

Each function with a prototype in `kleidicv.h` must be accompanied by benchmarks 
implemented in `benchmark/benchmark.cpp` based on
[Google Benchmark](https://github.com/google/benchmark). The image size to
execute the given operaion on can be set via the benchmark executable command
line (see [Benchmarking](doc/benchmark.md)). If there are other parameters of
the operation affecting performance then these parameters should be excercised
as well with representative inputs.

## Commit message

The subject line of the commit message should be 50 characters maximum and the
body should be wrapped at 72 characters. Please use imperative mood in the
subject line.

## Git hooks

To enable local commit message checks and checks for licenses, formatting, and
shell scripts, configure Git to use the repo's hooks directory:

```
git config core.hooksPath .githooks
```
