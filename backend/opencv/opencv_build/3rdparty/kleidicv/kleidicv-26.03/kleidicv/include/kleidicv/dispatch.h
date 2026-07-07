// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_DISPATCH_H
#define KLEIDICV_DISPATCH_H

#include "kleidicv/config.h"

#if KLEIDICV_ENABLE_SME2 || KLEIDICV_ENABLE_SME || KLEIDICV_ENABLE_SVE2

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace kleidicv {
#if KLEIDICV_ENABLE_SVE2
uint64_t svcntb_sve();
#endif
#if KLEIDICV_ENABLE_SME2 || KLEIDICV_ENABLE_SME
uint64_t svcntb_sme();
#endif
}  // namespace kleidicv

#ifdef __APPLE__

#include <sys/sysctl.h>

static inline bool query_sysctl(const char* attribute_name) {
  uint64_t attribute_value = 0;
  size_t max_attribute_size = sizeof(attribute_value);
  if (sysctlbyname(attribute_name, &attribute_value, &max_attribute_size, NULL,
                   0)) {
    return false;
  }
  return attribute_value;
}

static inline bool is_sve2_supported() {
  return query_sysctl("hw.optional.arm.FEAT_SVE2");
}

static inline bool is_sme_supported() {
  return query_sysctl("hw.optional.arm.FEAT_SME");
}

static inline bool is_sme2_supported() {
  return query_sysctl("hw.optional.arm.FEAT_SME2");
}

#else  // __APPLE__

#include <sys/auxv.h>

static inline bool is_sve2_supported() {
  return getauxval(AT_HWCAP2) & (1UL << 1);
}

static inline bool is_sme_supported() {
  return getauxval(AT_HWCAP2) & (1UL << 23);
}

static inline bool is_sme2_supported() {
  return getauxval(AT_HWCAP2) & (1UL << 37);
}

#endif  // __APPLE__

static inline bool is_prefer_sme_backend_env_var_set() {
  const char* v = std::getenv("KLEIDICV_PREFER_SME_BACKEND");
  if (v) {
    if (strcmp(v, "ON") == 0) {
      return true;
    }
  }
  return false;
}

#if KLEIDICV_ENABLE_SVE2
#define KLEIDICV_SVE2_RESOLVE(sve2_impl)                                     \
  if (!std::is_null_pointer_v<decltype(sve2_impl)> && is_sve2_supported()) { \
    return sve2_impl;                                                        \
  }
#define KLEIDICV_SVE2_RESOLVE_VECLEN(sve2_impl, sve_len)                     \
  if (!std::is_null_pointer_v<decltype(sve2_impl)> && is_sve2_supported() && \
      kleidicv::svcntb_sve() == (sve_len)) {                                 \
    return sve2_impl;                                                        \
  }
#else
#define KLEIDICV_SVE2_RESOLVE(x)
#define KLEIDICV_SVE2_RESOLVE_VECLEN(x, sve_len)
#endif  // KLEIDICV_ENABLE_SVE2

#if KLEIDICV_ENABLE_SME
#define KLEIDICV_SME_RESOLVE(sme_impl)                                     \
  if (!std::is_null_pointer_v<decltype(sme_impl)> && is_sme_supported()) { \
    return sme_impl;                                                       \
  }
#define KLEIDICV_SME_RESOLVE_VECLEN(sme_impl, sme_len)                     \
  if (!std::is_null_pointer_v<decltype(sme_impl)> && is_sme_supported() && \
      kleidicv::svcntb_sme() == (sme_len)) {                               \
    return sme_impl;                                                       \
  }
#else
#define KLEIDICV_SME_RESOLVE(x)
#define KLEIDICV_SME_RESOLVE_VECLEN(x, sme_len)
#endif  // KLEIDICV_ENABLE_SME

#if KLEIDICV_ENABLE_SME2
#define KLEIDICV_SME2_RESOLVE(sme2_impl)                                     \
  if (!std::is_null_pointer_v<decltype(sme2_impl)> && is_sme2_supported()) { \
    return sme2_impl;                                                        \
  }
#define KLEIDICV_SME2_RESOLVE_VECLEN(sme2_impl, sme_len)                     \
  if (!std::is_null_pointer_v<decltype(sme2_impl)> && is_sme2_supported() && \
      kleidicv::svcntb_sme() == (sme_len)) {                                 \
    return sme2_impl;                                                        \
  }
#else
#define KLEIDICV_SME2_RESOLVE(x)
#define KLEIDICV_SME2_RESOLVE_VECLEN(x, sme_len)
#endif  // KLEIDICV_ENABLE_SME2

#define KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(api_name, neon_impl, \
                                                sve2_impl)           \
  static decltype(neon_impl) api_name##_resolver() {                 \
    KLEIDICV_SVE2_RESOLVE(sve2_impl);                                \
    return neon_impl;                                                \
  }                                                                  \
  extern "C" {                                                       \
  decltype(neon_impl) api_name = api_name##_resolver();              \
  }

#define KLEIDICV_MULTIVERSION_C_API_WITH_SME(api_name, neon_impl, sve2_impl, \
                                             sme_impl, sme2_impl)            \
  static decltype(neon_impl) api_name##_resolver_default() {                 \
    if (is_prefer_sme_backend_env_var_set()) {                               \
      KLEIDICV_SME2_RESOLVE(sme2_impl);                                      \
      KLEIDICV_SME_RESOLVE(sme_impl);                                        \
    }                                                                        \
    KLEIDICV_SVE2_RESOLVE(sve2_impl);                                        \
    return neon_impl;                                                        \
  }                                                                          \
  static decltype(neon_impl) api_name##_resolver_for_sme() {                 \
    KLEIDICV_SME2_RESOLVE(sme2_impl);                                        \
    KLEIDICV_SME_RESOLVE(sme_impl);                                          \
    KLEIDICV_SVE2_RESOLVE(sve2_impl);                                        \
    return neon_impl;                                                        \
  }                                                                          \
  extern "C" {                                                               \
  decltype(neon_impl) api_name = api_name##_resolver_default();              \
  decltype(neon_impl) api_name##_sme = api_name##_resolver_for_sme();        \
  }

#define KLEIDICV_MULTIVERSION_C_API_VECLEN(                                \
    api_name, neon_impl, sve2_impl, sme_impl, sme2_impl, sve_len, sme_len) \
  static decltype(neon_impl) api_name##_resolver_default() {               \
    if (is_prefer_sme_backend_env_var_set()) {                             \
      KLEIDICV_SME2_RESOLVE_VECLEN(sme2_impl, sme_len);                    \
      KLEIDICV_SME_RESOLVE_VECLEN(sme_impl, sme_len);                      \
    }                                                                      \
    KLEIDICV_SVE2_RESOLVE_VECLEN(sve2_impl, sve_len);                      \
    return neon_impl;                                                      \
  }                                                                        \
  static decltype(neon_impl) api_name##_resolver_for_sme() {               \
    KLEIDICV_SME2_RESOLVE_VECLEN(sme2_impl, sme_len);                      \
    KLEIDICV_SME_RESOLVE_VECLEN(sme_impl, sme_len);                        \
    KLEIDICV_SVE2_RESOLVE_VECLEN(sve2_impl, sve_len);                      \
    return neon_impl;                                                      \
  }                                                                        \
  extern "C" {                                                             \
  decltype(neon_impl) api_name = api_name##_resolver_default();            \
  decltype(neon_impl) api_name##_sme = api_name##_resolver_for_sme();      \
  }

#else  // KLEIDICV_HAVE_SVE2 || KLEIDICV_HAVE_SME || KLEIDICV_HAVE_SME2

#define KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(api_name, neon_impl, \
                                                sve2_impl)           \
                                                                     \
  extern "C" {                                                       \
  decltype(neon_impl) api_name = neon_impl;                          \
  }

#define KLEIDICV_MULTIVERSION_C_API_WITH_SME(api_name, neon_impl, sve2_impl, \
                                             sme_impl, sme2_impl)            \
                                                                             \
  extern "C" {                                                               \
  decltype(neon_impl) api_name = neon_impl;                                  \
  decltype(neon_impl) api_name##_sme = neon_impl;                            \
  }

#define KLEIDICV_MULTIVERSION_C_API_VECLEN(                              \
    api_name, neon_impl, sve2_impl, sme_impl, sme2_impl, minlen, maxlen) \
  KLEIDICV_MULTIVERSION_C_API_WITH_SME(api_name, neon_impl, sve2_impl,   \
                                       sme_impl, sme2_impl)

#endif  // KLEIDICV_ENABLE_SME2 || KLEIDICV_ENABLE_SME ||  KLEIDICV_ENABLE_SVE2

#if KLEIDICV_ALWAYS_ENABLE_SME2
#define KLEIDICV_SME2_IMPL_IF(func) func
#else
#define KLEIDICV_SME2_IMPL_IF(func) nullptr
#endif  // KLEIDICV_ALWAYS_ENABLE_SME2

#if KLEIDICV_ALWAYS_ENABLE_SVE2
#define KLEIDICV_SVE2_IMPL_IF(func) func
#else
#define KLEIDICV_SVE2_IMPL_IF(func) nullptr
#endif  // KLEIDICV_ALWAYS_ENABLE_SVE2

#endif  // KLEIDICV_DISPATCH_H
