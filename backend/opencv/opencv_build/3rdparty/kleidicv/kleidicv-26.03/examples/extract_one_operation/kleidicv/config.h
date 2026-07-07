// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

// Static config based on kleidicv/include/kleidicv/config.h.in to meet the
// needs of the example.

#define KLEIDICV_LOCALLY_STREAMING __arm_locally_streaming
#define KLEIDICV_STREAMING __arm_streaming

#define KLEIDICV_UNLIKELY(cond) __builtin_expect((cond), 0)

#define KLEIDICV_ATTR_ALIGNED(alignment) __attribute__((aligned(alignment)))

#ifdef __clang__
#define KLEIDICV_FORCE_LOOP_UNROLL _Pragma("clang loop unroll(full)")
#else
// GCC doesn't have clang's unroll(full). 16 is typically plenty.
#define KLEIDICV_FORCE_LOOP_UNROLL _Pragma("GCC unroll 16")
#endif

#if defined(__clang__) || defined(__GNUC__)
#define KLEIDICV_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define KLEIDICV_FORCE_INLINE __forceinline
#else
#define KLEIDICV_FORCE_INLINE inline
#endif
