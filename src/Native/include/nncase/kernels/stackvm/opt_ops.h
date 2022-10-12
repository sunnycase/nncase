/* This file is generated by tools/stackvm_gen/IsaGen at 2022/5/14 下午6:47:34
 * +08:00.
 *
 * Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/tensor.h>
#include <nncase/value.h>

BEGIN_NS_NNCASE_KERNELS_MODULE(stackvm)
namespace optimized {

NNCASE_API result<void>
conv2d(const float *input, const float *weights, const float *bias,
       float *output, const dims_t &in_shape, const dims_t &in_strides,
       const dims_t &w_shape, NNCASE_UNUSED const dims_t &w_strides,
       NNCASE_UNUSED const dims_t &bias_strides,
       NNCASE_UNUSED const dims_t &out_strides, const padding &padding_h,
       const padding &padding_w, int32_t groups, int32_t stride_h,
       int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
       value_range<float> fused_activation,
       NNCASE_UNUSED kernels::kernel_context &context) noexcept;

NNCASE_API result<void>
gather_nd(datatype_t type, const gsl::byte *input, gsl::byte *output,
          const dims_t &in_shape, const dims_t &out_shape,
          const dims_t &in_strides, const dims_t &out_strides,
          datatype_t indices_type, const gsl::byte *indices,
          const dims_t &indices_shape, size_t batch_dims,
          kernel_context &context) noexcept;

NNCASE_API result<void> concat(datatype_t type,
                               gsl::span<const gsl::byte *const> inputs,
                               gsl::byte *output, const dims_t &out_shape,
                               gsl::span<const dims_t> in_strides,
                               const dims_t &out_strides, size_t axis,
                               const dims_t &concat_dims,
                               kernel_context &context) noexcept;

NNCASE_API result<void>
dequantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input,
           gsl::byte *output, const dims_t &in_shape,
           NNCASE_UNUSED const dims_t &in_strides,
           NNCASE_UNUSED const dims_t &out_strides, float scale, float bias,
           NNCASE_UNUSED kernel_context &context) noexcept;

NNCASE_API result<void>
gather(datatype_t type, const gsl::byte *input, gsl::byte *output,
       const dims_t &in_shape, const dims_t &out_shape,
       const dims_t &in_strides, const dims_t &out_strides,
       datatype_t indices_type, const gsl::byte *indices,
       const dims_t &indices_shape, size_t axis,
       kernel_context &context) noexcept;

NNCASE_API result<void>
one_hot(datatype_t type, datatype_t indices_type, const gsl::byte *indices,
        gsl::byte *output, const dims_t &indices_shape, const dims_t &out_shape,
        const dims_t &out_strides, size_t depth, gsl::byte *values, size_t axis,
        runtime::stackvm::one_hot_mode_t mode,
        kernel_context &context) noexcept;

NNCASE_API result<void>
quantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input,
         gsl::byte *output, const dims_t &in_shape,
         NNCASE_UNUSED const dims_t &in_strides,
         NNCASE_UNUSED const dims_t &out_strides, float scale, float bias,
         NNCASE_UNUSED kernel_context &context) noexcept;

NNCASE_API result<void>
resize_bilinear(typecode_t type, const gsl::byte *input, gsl::byte *output,
                const dims_t &in_shape, const dims_t &in_strides,
                const dims_t &out_strides, int32_t out_h, int32_t out_w,
                bool align_corners, bool half_pixel_centers,
                kernel_context &context) noexcept;

NNCASE_API result<void> resize_nearest_neighbor(
    typecode_t type, const gsl::byte *input, gsl::byte *output,
    const dims_t &in_shape, const dims_t &in_strides, const dims_t &out_strides,
    int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept;

NNCASE_API result<void>
slice(datatype_t type, const gsl::byte *input, gsl::byte *output,
      const dims_t &in_shape, const strides_t &in_strides,
      const strides_t &out_strides, const axes_t &begins, const axes_t &ends,
      const axes_t &strides, NNCASE_UNUSED kernel_context &context) noexcept;


result<void> binary(
    typecode_t typecode, runtime::stackvm::binary_op_t op, const gsl::byte *lhs,
    const gsl::byte *rhs, gsl::byte *output, const dims_t &lhs_shape,
    const strides_t &lhs_strides, const dims_t &rhs_shape,
    const strides_t &rhs_strides, const dims_t &out_shape,
    const strides_t &out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept;

NNCASE_API result<void> unary(typecode_t dtype, runtime::stackvm::unary_op_t op, const gsl::byte *in,
                 gsl::byte *out, const dims_t &shape,
                 const strides_t &in_strides, const dims_t &out_shape,
                 const strides_t &out_strides,
                 kernel_context &context = default_kernel_context()) noexcept;

//template <typename T>
//NNCASE_API result<void> matmul(const T *input_a, const T *input_b, const T *bias, T *output,
//                               const dims_t &in_a_shape, const dims_t &in_a_strides, const dims_t &in_b_shape,
//                               const dims_t &in_b_strides, const dims_t &out_shape, const dims_t &out_strides,
//                               value_range<float> fused_activation) noexcept;

template <typename T>
NNCASE_API result<void> softmax(const T *input, T *output, const dims_t &in_shape, const dims_t &in_strides,
                                const dims_t &out_strides, int32_t axis, float beta) noexcept;

template <typename T>
NNCASE_API result<void> sigmoid(const T *input, T *output, const dims_t &in_shape, const strides_t &input_strides, const dims_t &out_shape,               \
                                const strides_t &out_strides,
                                kernel_context &context = default_kernel_context()) noexcept;

} // namespace optimized
END_NS_NNCASE_KERNELS_MODULE
