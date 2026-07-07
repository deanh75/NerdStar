// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SCALE_SC_H
#define KLEIDICV_SCALE_SC_H

#include <algorithm>

#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

class AddFloat final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  explicit AddFloat(const svfloat32_t &svshift) KLEIDICV_STREAMING
      : svshift_{svshift} {}

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx, VectorType src) KLEIDICV_STREAMING {
    return svadd_x(ctx.predicate(), src, svshift_);
  }
  // NOLINTEND(readability-make-member-function-const)

 private:
  const svfloat32_t &svshift_;
};  // end of class AddFloat

class ScaleFloat final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  ScaleFloat(const svfloat32_t &svscale,
             const svfloat32_t &svshift) KLEIDICV_STREAMING
      : svscale_{svscale},
        svshift_{svshift} {}

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx, VectorType src) KLEIDICV_STREAMING {
    return svmla_x(ctx.predicate(), svshift_, src, svscale_);
  }
  // NOLINTEND(readability-make-member-function-const)

 private:
  const svfloat32_t &svscale_, &svshift_;
};  // end of class ScaleFloat

template <typename T, typename U>
static kleidicv_error_t scale_sc(const T *src, size_t src_stride, U *dst,
                                 size_t dst_stride, size_t width, size_t height,
                                 double scale, double shift) KLEIDICV_STREAMING;

// Specialization for float
template <>
kleidicv_error_t scale_sc(const float *src, size_t src_stride, float *dst,
                          size_t dst_stride, size_t width, size_t height,
                          double scale, double shift) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const float> src_rows{src, src_stride};
  Rows<float> dst_rows{dst, dst_stride};
  svfloat32_t svscale = svdup_f32(static_cast<float>(scale));
  svfloat32_t svshift = svdup_f32(static_cast<float>(shift));
  if (scale == 1.0) {
    AddFloat operation(svshift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    ScaleFloat operation(svscale, svshift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  }
  return KLEIDICV_OK;
}

// -----------------------------------------------------------------------
// Scale uint8 to float16
// -----------------------------------------------------------------------

class ScaleUint8ToFloat16Calc16 {
 public:
  using SrcType = uint8_t;
  using SrcVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<SrcType>;
  using SrcVectorType = typename SrcVecTraits::VectorType;
  using SrcVector2Type = typename SrcVecTraits::Vector2Type;
  using DstType = float16_t;
  using DstVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<DstType>;
  using DstVectorType = typename DstVecTraits::VectorType;
  using DstVector2Type = typename DstVecTraits::Vector2Type;
  using DstVector4Type = typename DstVecTraits::Vector4Type;

  ScaleUint8ToFloat16Calc16(double scale, double shift, svfloat16_t &svscale,
                            svfloat16_t &svshift) KLEIDICV_STREAMING
      : svscale_{svscale},
        svshift_{svshift} {
    svscale_ = svdup_n_f16(static_cast<float16_t>(scale));
    svshift_ = svdup_n_f16(static_cast<float16_t>(shift));
  }

  void process_row(size_t width, Columns<const SrcType> src,
                   Columns<DstType> dst) const KLEIDICV_STREAMING {
    svbool_t p16 = svptrue_b16();
    svuint8_t svzero = svdup_n_u8(0);
#if KLEIDICV_TARGET_SME2
    svcount_t pc8 = SrcVecTraits::svptrue_c();
    svcount_t pc16 = DstVecTraits::svptrue_c();
#else
    svbool_t p8 = svptrue_b8();
#endif
    auto vector_path = [&](svuint16_t src) KLEIDICV_STREAMING {
      svfloat16_t fsrc = svcvt_f16_x(p16, src);
      return svmla_f16_x(p16, svshift_, fsrc, svscale_);
    };

    LoopUnroll{width, SrcVecTraits::num_lanes()}
        .unroll_twice([&](size_t step) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
          SrcVector2Type src_2vec = svld1_x2(pc8, &src[0]);
          svuint8_t src0 = svget2(src_2vec, 0);
          svuint8_t src1 = svget2(src_2vec, 1);
#else
          svuint8_t src0 = svld1(p8, &src[0]);
          svuint8_t src1 = svld1_vnum(p8, &src[0], 1);
#endif  // KLEIDICV_TARGET_SME2
          DstVectorType dst0 =
              vector_path(svreinterpret_u16_u8(svzip1(src0, svzero)));
          DstVectorType dst1 =
              vector_path(svreinterpret_u16_u8(svzip2(src0, svzero)));
          DstVectorType dst2 =
              vector_path(svreinterpret_u16_u8(svzip1(src1, svzero)));
          DstVectorType dst3 =
              vector_path(svreinterpret_u16_u8(svzip2(src1, svzero)));
#if KLEIDICV_TARGET_SME2
          DstVector4Type dst4 = svcreate4(dst0, dst1, dst2, dst3);
          svst1(pc16, &dst[0], dst4);
#else
          svst1(p16, &dst[0], dst0);
          svst1_vnum(p16, &dst[0], 1, dst1);
          svst1_vnum(p16, &dst[0], 2, dst2);
          svst1_vnum(p16, &dst[0], 3, dst3);
#endif  // KLEIDICV_TARGET_SME2
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) KLEIDICV_STREAMING {
          disable_loop_vectorization();
          size_t step = svcnth();
          while (length > 0) {
            svbool_t p16 = svwhilelt_b16_u64(0, length);
            svuint16_t src0 = svld1ub_u16(p16, &src[0]);
            DstVectorType dst0 = vector_path(src0);
            svst1(p16, &dst[0], dst0);
            src += ptrdiff_t(step);
            dst += ptrdiff_t(step);
            length -= std::min<size_t>(length, step);
          }
        });
  }

  svfloat16_t &svscale_, &svshift_;
};  // end of class ScaleUint8ToFloat16Calc16

class ScaleUint8ToFloat16Calc32 {
 public:
  using SrcType = uint8_t;
  using SrcVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<SrcType>;
  using SrcVectorType = typename SrcVecTraits::VectorType;
  using SrcVector2Type = typename SrcVecTraits::Vector2Type;
  using DstType = float16_t;
  using DstVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<DstType>;
  using DstVectorType = typename DstVecTraits::VectorType;
  using DstVector2Type = typename DstVecTraits::Vector2Type;
  using DstVector4Type = typename DstVecTraits::Vector4Type;

  ScaleUint8ToFloat16Calc32(double scale, double shift, svfloat32_t &svscale,
                            svfloat32_t &svshift) KLEIDICV_STREAMING
      : svscale_{svscale},
        svshift_{svshift} {
    svscale_ = svdup_n_f32(static_cast<float>(scale));
    svshift_ = svdup_n_f32(static_cast<float>(shift));
  }

  void process_row(size_t width, Columns<const SrcType> src,
                   Columns<DstType> dst) const KLEIDICV_STREAMING {
    svbool_t p16 = svptrue_b16();
    svbool_t p32 = svptrue_b32();
    svuint8_t svzero = svdup_n_u8(0);
#if KLEIDICV_TARGET_SME2
    svcount_t pc8 = SrcVecTraits::svptrue_c();
    svcount_t pc16 = DstVecTraits::svptrue_c();
#else
    svbool_t p8 = svptrue_b8();
#endif
    auto vector_path = [&](svuint8_t src) KLEIDICV_STREAMING {
      // First Transpose them, to have even and odd elements separated
      // so the final narrowing puts them into the right order.
      svuint8_t src_b = svtrn1(src, svzero);
      svuint8_t src_t = svtrn2(src, svzero);
      svuint8_t src_b0 = svzip1(src_b, svzero);
      svuint8_t src_b1 = svzip2(src_b, svzero);
      svuint8_t src_t0 = svzip1(src_t, svzero);
      svuint8_t src_t1 = svzip2(src_t, svzero);
      svfloat32_t fsrc_b0 = svcvt_f32_x(p32, svreinterpret_u32_u8(src_b0));
      svfloat32_t fsrc_b1 = svcvt_f32_x(p32, svreinterpret_u32_u8(src_b1));
      svfloat32_t fsrc_t0 = svcvt_f32_x(p32, svreinterpret_u32_u8(src_t0));
      svfloat32_t fsrc_t1 = svcvt_f32_x(p32, svreinterpret_u32_u8(src_t1));
      svfloat32_t res_b0 = svmla_f32_x(p32, svshift_, fsrc_b0, svscale_);
      svfloat32_t res_b1 = svmla_f32_x(p32, svshift_, fsrc_b1, svscale_);
      svfloat32_t res_t0 = svmla_f32_x(p32, svshift_, fsrc_t0, svscale_);
      svfloat32_t res_t1 = svmla_f32_x(p32, svshift_, fsrc_t1, svscale_);
      svfloat16_t res0 = svcvtnt_f16_x(svcvt_f16_x(p16, res_b0), p16, res_t0);
      svfloat16_t res1 = svcvtnt_f16_x(svcvt_f16_x(p16, res_b1), p16, res_t1);
      return svcreate2(res0, res1);
    };

    LoopUnroll{width, SrcVecTraits::num_lanes()}
        .unroll_twice([&](size_t step) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
          SrcVector2Type src_2vec = svld1_x2(pc8, &src[0]);
          svuint8_t src0 = svget2(src_2vec, 0);
          svuint8_t src1 = svget2(src_2vec, 1);
#else
          svuint8_t src0 = svld1(p8, &src[0]);
          svuint8_t src1 = svld1_vnum(p8, &src[0], 1);
#endif  // KLEIDICV_TARGET_SME2
          DstVector2Type dst0 = vector_path(src0);
          DstVector2Type dst1 = vector_path(src1);
#if KLEIDICV_TARGET_SME2
          DstVector4Type dst4 = svcreate4(svget2(dst0, 0), svget2(dst0, 1),
                                          svget2(dst1, 0), svget2(dst1, 1));
          svst1(pc16, &dst[0], dst4);
#else
          svst1(p16, &dst[0], svget2(dst0, 0));
          svst1_vnum(p16, &dst[0], 1, svget2(dst0, 1));
          svst1_vnum(p16, &dst[0], 2, svget2(dst1, 0));
          svst1_vnum(p16, &dst[0], 3, svget2(dst1, 1));
#endif  // KLEIDICV_TARGET_SME2
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) KLEIDICV_STREAMING {
          size_t step = svcnth();
          while (length > 0) {
            disable_loop_vectorization();
            svbool_t p16 = svwhilelt_b16_u64(0, length);
            svuint16_t src0 = svld1ub_u16(p16, &src[0]);
            svuint16_t src_b = svtrn1(src0, svreinterpret_u16_u8(svzero));
            svuint16_t src_t = svtrn2(src0, svreinterpret_u16_u8(svzero));
            svfloat32_t fsrc_b = svcvt_f32_x(p32, svreinterpret_u32_u16(src_b));
            svfloat32_t fsrc_t = svcvt_f32_x(p32, svreinterpret_u32_u16(src_t));
            svfloat32_t res_b = svmla_f32_x(p32, svshift_, fsrc_b, svscale_);
            svfloat32_t res_t = svmla_f32_x(p32, svshift_, fsrc_t, svscale_);
            svfloat16_t res =
                svcvtnt_f16_x(svcvt_f16_x(p16, res_b), p16, res_t);
            svst1(p16, &dst[0], res);
            src += ptrdiff_t(step);
            dst += ptrdiff_t(step);
            length -= std::min<size_t>(length, step);
          }
        });
  }

  svfloat32_t &svscale_, &svshift_;
};  // end of class ScaleUint8ToFloat16Calc32

// Specialization for uint8_t to float16_t
template <>
kleidicv_error_t scale_sc(const uint8_t *src, size_t src_stride, float16_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          double scale, double shift) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride};
  Rows<float16_t> dst_rows{dst, dst_stride};
  if (static_cast<double>(static_cast<float16_t>(scale)) == scale &&
      static_cast<double>(static_cast<float16_t>(shift)) == shift) {
    svfloat16_t s0, s1;
    ScaleUint8ToFloat16Calc16 operation(scale, shift, s0, s1);
    zip_rows(operation, rect, src_rows, dst_rows);
  } else {
    svfloat32_t s0, s1;
    ScaleUint8ToFloat16Calc32 operation(scale, shift, s0, s1);
    zip_rows(operation, rect, src_rows, dst_rows);
  }

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SCALE_SC_H
