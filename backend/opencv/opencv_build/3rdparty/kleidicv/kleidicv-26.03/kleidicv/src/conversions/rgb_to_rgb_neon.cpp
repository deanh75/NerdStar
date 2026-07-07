// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/traits.h"
#include "kleidicv/types.h"

namespace kleidicv::neon {

template <typename ScalarType>
class RGBToBGR final {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  KLEIDICV_FORCE_INLINE
  void process_row(size_t length, Columns<const uint8_t> src,
                   Columns<uint8_t> dst) {
    LoopUnroll loop{length, 16};
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    uint8x16x3_t indices;
    VecTraits::load(kRGBToBGRTableIndices, indices);
#endif  // !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

    loop.unroll_once([&](size_t step) {
      KLEIDICV_PREFETCH(&src[0] + 1024);
      uint8x16x3_t dst_vect;
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE || defined(__clang__)
      uint8x16x3_t src_vect;
#endif  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE || defined(__clang__)

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
      src_vect = vld3q(&src[0]);
      std::swap(src_vect.val[0], src_vect.val[2]);
      vst3q(&dst[0], dst_vect);
#else  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
#if defined(__clang__)
      VecTraits::load(&src[0], src_vect);

      dst_vect.val[0] =
          vqtbl2q_u8({src_vect.val[0], src_vect.val[1]}, indices.val[0]);
      dst_vect.val[1] = vqtbl3q_u8(
          {src_vect.val[0], src_vect.val[1], src_vect.val[2]}, indices.val[1]);
      dst_vect.val[2] =
          vqtbl2q_u8({src_vect.val[1], src_vect.val[2]}, indices.val[2]);
#else   // defined(__clang__)
      asm volatile(
          "ld1 { v16.16b, v17.16b, v18.16b }, [%[src_ptr]] \n\t"
          "tbl %0.16b, { v16.16b, v17.16b  }, %[idx0].16b \n\t"
          "tbl %1.16b, { v16.16b, v17.16b, v18.16b }, %[idx1].16b \n\t"
          "tbl %2.16b, { v17.16b, v18.16b }, %[idx2].16b \n\t"
          : "=&w"(dst_vect.val[0]), "=&w"(dst_vect.val[1]),
            "=&w"(dst_vect.val[2])
          : [src_ptr] "r"(&src[0]), [idx0] "w"(indices.val[0]),
            [idx1] "w"(indices.val[1]), [idx2] "w"(indices.val[2])
          : "v16", "v17", "v18", "memory");
#endif  // defined(__clang__)
      VecTraits::store(dst_vect, &dst[0]);
#endif  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
      src += static_cast<ptrdiff_t>(step);
      dst += static_cast<ptrdiff_t>(step);
    });

    loop.remaining([&](size_t length, size_t /* step */) {
      for (size_t index = 0; index < length; ++index) {
        disable_loop_vectorization();
        scalar_path(&src.at(static_cast<ptrdiff_t>(index))[0],
                    &dst.at(static_cast<ptrdiff_t>(index))[0]);
      }
    });
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  static constexpr uint8_t kRGBToBGRTableIndices[48] = {
      2,  1,  0,  5,  4,  3,  8,  7,  6,  11, 10, 9,  14, 13, 12, 17,
      16, 15, 20, 19, 18, 23, 22, 21, 26, 25, 24, 29, 28, 27, 32, 31,
      14, 19, 18, 17, 22, 21, 20, 25, 24, 23, 28, 27, 26, 31, 30, 29};
#endif  //  !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
  }
};  // end of class RGBToBGR<ScalarType>

template <typename ScalarType>
class RGBAToBGRA final : public UnrollOnce {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  RGBAToBGRA() : indices_{} {
    VecTraits::load(kRGBAToBGRATableIndices, indices_);
  }
#else
  RGBAToBGRA() = default;
#endif

  void vector_path(const ScalarType *src, ScalarType *dst) {
    KLEIDICV_PREFETCH(&src[0] + 1024);
    uint8x16x4_t src_vect, dst_vect;

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    src_vect = vld4q_u8(src);

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];
    dst_vect.val[3] = src_vect.val[3];

    vst4q_u8(dst, dst_vect);
#else
    VecTraits::load(&src[0], src_vect);

    dst_vect.val[0] = vqtbl1q_u8(src_vect.val[0], indices_);
    dst_vect.val[1] = vqtbl1q_u8(src_vect.val[1], indices_);
    dst_vect.val[2] = vqtbl1q_u8(src_vect.val[2], indices_);
    dst_vect.val[3] = vqtbl1q_u8(src_vect.val[3], indices_);

    VecTraits::store(dst_vect, &dst[0]);
#endif
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
    dst[3] = src[3];
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  static constexpr uint8_t kRGBAToBGRATableIndices[16] = {
      2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15};
  uint8x16_t indices_;
#endif
};  // end of class RGBAToBGRA<ScalarType>

template <typename ScalarType>
class RGBToBGRA final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x3_t src_vect = vld3q_u8(src);
    uint8x16x4_t dst_vect;

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];
    dst_vect.val[3] = vdupq_n_u8(0xff);

    vst4q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
    dst[3] = 0xff;
  }
};  // end of class RGBToBGRA<ScalarType>

template <typename ScalarType>
class RGBToRGBA final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x3_t src_vect = vld3q_u8(src);
    uint8x16x4_t dst_vect;

    dst_vect.val[0] = src_vect.val[0];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[2];
    dst_vect.val[3] = vdupq_n_u8(0xff);

    vst4q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    memcpy(static_cast<void *>(dst), static_cast<const void *>(src), 3);
    dst[3] = 0xff;
  }
};  // end of class RGBToRGBA<ScalarType>

template <typename ScalarType>
class RGBAToBGR final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x4_t src_vect = vld4q_u8(src);
    uint8x16x3_t dst_vect;

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];

    vst3q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
  }
};  // end of class RGBAToBGR<ScalarType>

template <typename ScalarType>
class RGBAToRGB final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x4_t src_vect = vld4q_u8(src);
    uint8x16x3_t dst_vect;

    dst_vect.val[0] = src_vect.val[0];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[2];

    vst3q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    memcpy(static_cast<void *>(dst), static_cast<const void *>(src), 3);
  }
};  // end of class RGBAToRGB<ScalarType>

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
  RGBToBGR<uint8_t> operation;
  zip_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* BGRA */};
  RGBAToBGRA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_bgra_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* BGRA */};
  RGBToBGRA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_rgba_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* RGBA */};
  RGBToRGBA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
  RGBAToBGR<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* RGB */};
  RGBAToRGB<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
